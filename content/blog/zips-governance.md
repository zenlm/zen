---
title: "ZIPs: Decentralized Governance for Open AI"
date: "2024-05-20"
description: "How Zoo Improvement Proposals enable community-driven governance of open AI development."
author: "Zach Kelling"
tags: ["Governance", "ZIPs"]
---


Since launching Zoo Labs Foundation, we've processed 47 Zoo Improvement Proposals (ZIPs). Today we share lessons learned and improvements to the governance process.

## The Case for Governance

AI development involves decisions that affect many stakeholders:

- What data should models train on?
- How should capabilities be released?
- What safety measures are required?
- How should resources be allocated?

These questions don't have purely technical answers. They require value judgments. Centralized organizations make these judgments internally. We believe affected communities should participate.

ZIPs make governance explicit, transparent, and participatory.

## How ZIPs Work

### Proposal Types

ZIPs fall into several categories:

**Standard ZIPs**: Technical specifications
- Model architectures
- Training procedures
- Evaluation protocols
- API standards

**Process ZIPs**: Governance procedures
- Voting rules
- Working group charters
- Grant criteria
- Release policies

**Informational ZIPs**: Guidelines and recommendations
- Best practices
- Research directions
- Community standards

### Lifecycle

```
[Draft] -> [Review] -> [Vote] -> [Accepted/Rejected] -> [Implemented]
                |
                v
           [Withdrawn]
```

**Draft**: Author creates proposal using template. Must include:
- Abstract (200 words max)
- Motivation (why this change?)
- Specification (precise details)
- Rationale (why this approach?)
- Backwards compatibility
- Security considerations

**Review**: Community discusses on forums and in working groups. Minimum 14 days. Author revises based on feedback.

**Vote**: Token holders vote. Quorum: 10% of circulating supply. Threshold: 66% approval for Standard/Process ZIPs, 50% for Informational.

**Implementation**: Approved ZIPs are implemented by relevant teams. Timeline specified in proposal.

### Example ZIP

**ZIP-23: Mandatory Safety Evaluations**

*Abstract*: This ZIP requires all Zen models to pass standardized safety evaluations before release. It specifies evaluation benchmarks, minimum thresholds, and remediation procedures.

*Motivation*: As Zen models become more capable, systematic safety evaluation ensures we catch potential harms before deployment...

[Full proposal at zips.zoo.ngo/zip-23]

## Lessons Learned

After 47 ZIPs, we've learned what works and what doesn't.

### What Works

**Clear templates**: Structured templates reduce ambiguity. Authors know what to include. Reviewers know what to expect.

**Mandatory discussion periods**: Two weeks minimum prevents rushed decisions. Complex proposals often need longer.

**Working group pre-review**: Technical working groups review proposals before community vote. Catches errors and improves quality.

**On-chain voting**: Tamper-proof voting records. Clear outcome determination. Historical audit trail.

### What Doesn't Work

**Low participation**: Early votes had <5% participation. We added delegation and reminders. Participation now averages 12%.

**Proposal complexity**: Some proposals tried to do too much. We now encourage smaller, focused ZIPs.

**Implementation lag**: Approved proposals sometimes stalled in implementation. We added accountability mechanisms.

**Voter fatigue**: Too many votes led to declining participation. We batch non-urgent votes monthly.

## Improvements for 2024

Based on experience, we're upgrading the process:

### Conviction Voting

For resource allocation decisions, we're piloting conviction voting:

- Tokens continuously signal preferences
- Conviction accumulates over time
- Proposals pass when conviction reaches threshold

This rewards sustained support over flash mobs.

### Optimistic Approval

Low-controversy changes use optimistic approval:

- Proposal published with 7-day objection period
- If no objection reaches threshold (5% of tokens), proposal passes
- If objection threshold reached, standard vote occurs

Reduces voting overhead for routine decisions.

### Working Group Autonomy

Working groups can approve changes within their scope without full vote:

- Changes must align with approved charter
- Community can override within 14 days
- Increases agility while maintaining accountability

### Reputation-Weighted Voting

We're exploring reputation adjustments:

- Technical expertise weighted higher for technical ZIPs
- Contribution history affects weight
- Prevents pure plutocracy

This is controversial. ZIP-52 is the formal proposal for community decision.

## Governance Statistics

| Metric | Q1 2024 | Q2 2024 |
|--------|---------|---------|
| Proposals submitted | 18 | 29 |
| Proposals passed | 12 | 21 |
| Average participation | 8% | 14% |
| Average discussion length | 9 days | 16 days |
| Implementation rate | 75% | 92% |

Participation and quality are improving.

## Notable ZIPs

Some highlights from our proposal history:

**ZIP-7**: Established the grants program criteria and evaluation process. Enabled $3M in grants.

**ZIP-15**: Required training data documentation for all Zen models. Now standard practice.

**ZIP-23**: Mandatory safety evaluations (discussed above). Caught two issues pre-release.

**ZIP-31**: Created the Infrastructure Working Group. Now maintains Training Gym.

**ZIP-38**: Approved Zen-2-70B training budget. Largest single resource allocation.

## Participating in Governance

### For Token Holders

1. **Stay informed**: Read proposals at zips.zoo.ngo
2. **Discuss**: Join forum discussions
3. **Delegate**: If you can't follow closely, delegate to aligned representatives
4. **Vote**: Participate in votes that matter to you

### For Proposers

1. **Start with discussion**: Float ideas in forums before formal proposal
2. **Find co-sponsors**: Working group endorsement helps
3. **Be specific**: Vague proposals fail
4. **Engage with feedback**: Responsive authors succeed

### For Working Groups

1. **Pre-review proposals**: Catch issues early
2. **Provide expertise**: Technical analysis helps voters
3. **Implement promptly**: Approved proposals deserve attention

## Governance as Infrastructure

Good governance is invisible when working. Bad governance is obvious when failing. ZIPs provide structure that enables coordinated action while preventing unilateral capture.

As AI systems become more powerful, their governance becomes more important. We're building the institutions for a future where AI development is genuinely participatory.

Join the conversation at zips.zoo.ngo.

---

*Zach Kelling is a co-founder of Zoo Labs Foundation.*
