"""Project 9 - Multi-Agent Debate System.

Round 0: N persona agents propose solutions independently (in parallel).
Round k: each agent sees the others' proposals + critic feedback and may revise.
Critic : scores every proposal on a rubric.
Voting : every agent ranks the *other* proposals (Borda count, no self-votes),
         combined with critic scores -> consensus check (quorum) -> early stop.
Aggregator: synthesizes the final answer with a calibrated confidence built
         from agreement, critic quality and the agents' own confidence.
"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from pydantic import BaseModel, Field

from common.llm import BaseLLM
from common.structured import StructuredOutputError, generate_structured


@dataclass(frozen=True)
class Persona:
    name: str
    perspective: str


DEFAULT_PERSONAS = [
    Persona("risk_analyst", "You minimise fraud losses and think in terms of expected loss and attack vectors."),
    Persona("customer_advocate", "You protect customer experience; false positives and friction are costly."),
    Persona("compliance_officer", "You ensure regulatory compliance (RBI, KYC/AML), auditability and fairness."),
]


class Proposal(BaseModel):
    solution: str = Field(description="Concrete recommended solution")
    key_arguments: list[str] = Field(min_length=1, max_length=5)
    confidence: float = Field(ge=0, le=1)


class ProposalScore(BaseModel):
    proposal_id: str
    correctness: int = Field(ge=1, le=10)
    feasibility: int = Field(ge=1, le=10)
    risk_awareness: int = Field(ge=1, le=10)
    critique: str


class CriticReport(BaseModel):
    scores: list[ProposalScore]


class Ballot(BaseModel):
    ranking: list[str] = Field(description="Proposal ids from best to worst (exclude your own)")
    rationale: str


class FinalAnswer(BaseModel):
    answer: str
    dissent: list[str] = Field(default_factory=list, description="Important minority objections to keep")


@dataclass
class RoundRecord:
    round: int
    proposals: dict[str, dict]
    critic: dict[str, dict]
    borda: dict[str, float]
    combined: dict[str, float]
    leader: str
    consensus: float


@dataclass
class DebateResult:
    question: str
    answer: str
    confidence: float
    winner: str
    consensus_reached: bool
    rounds: list[RoundRecord] = field(default_factory=list)
    dissent: list[str] = field(default_factory=list)


class DebateSystem:
    def __init__(self, llm: BaseLLM, personas: list[Persona] | None = None, max_rounds: int = 3,
                 quorum: float = 0.67, critic_weight: float = 0.5):
        self.llm = llm
        self.personas = personas or DEFAULT_PERSONAS
        self.max_rounds = max_rounds
        self.quorum = quorum
        self.critic_weight = critic_weight

    # ---------------------------------------------------------------- agents
    def _propose(self, p: Persona, question: str, others: dict[str, dict] | None, feedback: dict | None) -> Proposal:
        msgs = [{"role": "system", "content": f"You are {p.name}. {p.perspective} Propose a solution."},
                {"role": "user", "content": f"Problem: {question}"}]
        if others:
            msgs.append({"role": "user", "content": (
                "Other agents proposed:\n" + json.dumps(others, indent=1) +
                f"\nCritic feedback on your previous proposal: {json.dumps(feedback or {})}\n"
                "Revise your proposal: keep what is right, adopt better ideas from others, defend your view "
                "only where you have strong arguments.")})
        return generate_structured(self.llm, msgs, Proposal, max_retries=1, tag=f"p09.propose.{p.name}").value

    def _critic(self, question: str, proposals: dict[str, dict]) -> dict[str, dict]:
        msgs = [{"role": "system", "content": "You are an impartial expert critic. Score each proposal 1-10 on "
                                              "correctness, feasibility and risk_awareness, with a one-line critique."},
                {"role": "user", "content": f"Problem: {question}\nProposals:\n{json.dumps(proposals, indent=1)}"}]
        try:
            rep = generate_structured(self.llm, msgs, CriticReport, max_retries=1, tag="p09.critic").value
        except StructuredOutputError:
            return {}
        return {s.proposal_id: s.model_dump() for s in rep.scores if s.proposal_id in proposals}

    def _vote(self, p: Persona, question: str, proposals: dict[str, dict]) -> list[str]:
        others = {k: v for k, v in proposals.items() if k != p.name}
        msgs = [{"role": "system", "content": f"You are {p.name}. Rank the OTHER agents' proposals from best to worst "
                                              "on merit, not on whether they agree with you."},
                {"role": "user", "content": f"Problem: {question}\nProposals:\n{json.dumps(others, indent=1)}"}]
        try:
            ballot = generate_structured(self.llm, msgs, Ballot, max_retries=1, tag=f"p09.vote.{p.name}").value
        except StructuredOutputError:
            return []
        seen, ranking = set(), []
        for pid in ballot.ranking:  # sanitize: no self-votes, no unknown ids, no duplicates
            if pid in others and pid not in seen:
                ranking.append(pid)
                seen.add(pid)
        return ranking

    # ---------------------------------------------------------------- scoring
    @staticmethod
    def borda(ballots: dict[str, list[str]], candidates: list[str]) -> dict[str, float]:
        """Normalised Borda count in [0,1]: share of the maximum points a candidate could get."""
        points = {c: 0.0 for c in candidates}
        maxpts = {c: 0.0 for c in candidates}
        for voter, ranking in ballots.items():
            eligible = [c for c in candidates if c != voter]
            n = len(eligible)
            for c in eligible:
                maxpts[c] += n - 1
            for pos, c in enumerate(ranking):
                points[c] += n - 1 - pos
        return {c: (points[c] / maxpts[c]) if maxpts[c] else 0.0 for c in candidates}

    @staticmethod
    def critic_norm(s: dict | None) -> float:
        if not s:
            return 0.5
        return (s["correctness"] + s["feasibility"] + s["risk_awareness"] - 3) / 27

    def _parallel(self, fn, items):
        with ThreadPoolExecutor(max_workers=len(items)) as ex:
            return list(ex.map(fn, items))

    # ---------------------------------------------------------------- main
    def run(self, question: str) -> DebateResult:
        names = [p.name for p in self.personas]
        proposals: dict[str, dict] = {}
        critic: dict[str, dict] = {}
        rounds: list[RoundRecord] = []
        for rnd in range(self.max_rounds):
            prev = dict(proposals)

            def propose(p: Persona):
                others = {k: v for k, v in prev.items() if k != p.name} if prev else None
                try:
                    return p.name, self._propose(p, question, others, critic.get(p.name)).model_dump()
                except StructuredOutputError:
                    return p.name, prev.get(p.name)
            proposals = {k: v for k, v in self._parallel(propose, self.personas) if v}
            if not proposals:
                break
            critic = self._critic(question, proposals)
            ballots = dict(self._parallel(lambda p: (p.name, self._vote(p, question, proposals)),
                                          [p for p in self.personas if p.name in proposals]))
            cands = list(proposals)
            b = self.borda(ballots, cands)
            combined = {c: round((1 - self.critic_weight) * b[c] + self.critic_weight * self.critic_norm(critic.get(c)), 3)
                        for c in cands}
            leader = max(combined, key=combined.get)
            # consensus = share of voters (other than the leader) whose top choice is the leader
            voters = [v for v, r in ballots.items() if v != leader and r]
            consensus = sum(1 for v in voters if ballots[v][0] == leader) / len(voters) if voters else 0.0
            rounds.append(RoundRecord(rnd, proposals, critic, {k: round(v, 3) for k, v in b.items()}, combined,
                                      leader, round(consensus, 3)))
            if consensus >= self.quorum:
                break

        last = rounds[-1] if rounds else None
        if last is None:
            return DebateResult(question, "No proposals could be generated.", 0.0, "", False)
        winner = last.leader
        try:
            final = generate_structured(self.llm, [
                {"role": "system", "content": "You are the aggregator. Synthesize the best final answer: build on the "
                                              "winning proposal, merge strong points from others, and list important "
                                              "dissenting objections."},
                {"role": "user", "content": json.dumps({"problem": question, "winner": winner, "proposals": last.proposals,
                                                        "critic": last.critic, "scores": last.combined})}],
                FinalAnswer, max_retries=1, tag="p09.aggregate").value
        except StructuredOutputError:
            final = FinalAnswer(answer=last.proposals[winner]["solution"])
        avg_conf = sum(p["confidence"] for p in last.proposals.values()) / len(last.proposals)
        confidence = round(0.4 * last.consensus + 0.35 * self.critic_norm(last.critic.get(winner)) + 0.25 * avg_conf, 3)
        return DebateResult(question, final.answer, confidence, winner, last.consensus >= self.quorum, rounds,
                            final.dissent)
