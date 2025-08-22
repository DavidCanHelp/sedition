---------------------------- MODULE ProofOfContribution ----------------------------
(* 
 * TLA+ Formal Specification of Proof of Contribution Consensus
 * This specification formally verifies safety and liveness properties
 *)

EXTENDS Integers, Sequences, FiniteSets, TLC

CONSTANTS 
    Validators,           \* Set of validator identities
    MaxRounds,           \* Maximum number of consensus rounds
    MinQuality,          \* Minimum quality threshold for contributions
    ByzantineValidators  \* Set of Byzantine validators

ASSUME 
    /\ Cardinality(ByzantineValidators) < Cardinality(Validators) \div 3
    /\ ByzantineValidators \subseteq Validators
    /\ MinQuality \in 0..100
    /\ MaxRounds \in Nat \ {0}

VARIABLES
    round,               \* Current consensus round
    stake,               \* Function mapping validators to stake amounts
    reputation,          \* Function mapping validators to reputation scores
    contributions,       \* Function mapping validators to contribution scores
    blocks,              \* Sequence of accepted blocks
    proposals,           \* Current round proposals
    votes,               \* Votes for current round
    decided,             \* Whether consensus has been reached
    messages             \* Network messages

vars == <<round, stake, reputation, contributions, blocks, proposals, votes, decided, messages>>

-------------------------------------------------------------------------------------
\* Type Invariants
-------------------------------------------------------------------------------------

TypeOK ==
    /\ round \in 0..MaxRounds
    /\ stake \in [Validators -> Nat]
    /\ reputation \in [Validators -> 0..100]
    /\ contributions \in [Validators -> 0..100]
    /\ blocks \in Seq([proposer: Validators, quality: 0..100, round: 0..MaxRounds])
    /\ proposals \in SUBSET [proposer: Validators, block: STRING, quality: 0..100]
    /\ votes \in [Validators -> [block: STRING, round: 0..MaxRounds]]
    /\ decided \in BOOLEAN
    /\ messages \in SUBSET [from: Validators, to: Validators, type: STRING, content: STRING]

-------------------------------------------------------------------------------------
\* Helper Functions
-------------------------------------------------------------------------------------

\* Calculate total stake for a validator
TotalStake(v) ==
    LET repMultiplier == (reputation[v] \div 10) + 1
        contribBonus == 1 + (contributions[v] \div 200)
    IN stake[v] * repMultiplier * contribBonus

\* Calculate total network stake
NetworkStake == 
    LET validatorSet == DOMAIN stake
    IN Sum(v \in validatorSet, TotalStake(v))

\* Sum helper function
Sum(S, f(_)) == 
    LET F[s \in SUBSET S] == 
        IF s = {} THEN 0
        ELSE LET x == CHOOSE x \in s : TRUE
             IN f(x) + F[s \ {x}]
    IN F[S]

\* Check if validator is honest
IsHonest(v) == v \notin ByzantineValidators

\* Calculate voting power of a set of validators
VotingPower(S) ==
    Sum(S, TotalStake)

\* Select leader based on VRF (simplified as deterministic for model checking)
SelectLeader(r) ==
    CHOOSE v \in Validators : 
        \A w \in Validators \ {v} : 
            (TotalStake(v) * ((v + r) % 100)) >= (TotalStake(w) * ((w + r) % 100))

\* Check if a block has sufficient votes
HasSuperMajority(block) ==
    LET supporters == {v \in Validators : votes[v].block = block}
    IN VotingPower(supporters) > (2 * NetworkStake) \div 3

-------------------------------------------------------------------------------------
\* Initial State
-------------------------------------------------------------------------------------

Init ==
    /\ round = 0
    /\ stake \in [Validators -> 100..1000]      \* Random initial stakes
    /\ reputation = [v \in Validators |-> 50]   \* Start with neutral reputation
    /\ contributions = [v \in Validators |-> 50] \* Average initial contributions
    /\ blocks = <<>>
    /\ proposals = {}
    /\ votes = [v \in Validators |-> [block |-> "none", round |-> 0]]
    /\ decided = FALSE
    /\ messages = {}

-------------------------------------------------------------------------------------
\* Actions
-------------------------------------------------------------------------------------

\* Honest validator proposes a block
ProposeBlock(v) ==
    /\ IsHonest(v)
    /\ v = SelectLeader(round)
    /\ round < MaxRounds
    /\ ~decided
    /\ LET quality == contributions[v]
           newBlock == [proposer |-> v, block |-> ToString(round), quality |-> quality]
       IN /\ quality >= MinQuality
          /\ proposals' = proposals \cup {newBlock}
          /\ UNCHANGED <<round, stake, reputation, contributions, blocks, votes, decided, messages>>

\* Byzantine validator proposes a malicious block
ProposeMaliciousBlock(v) ==
    /\ ~IsHonest(v)
    /\ v = SelectLeader(round)
    /\ round < MaxRounds
    /\ ~decided
    /\ LET quality == 0  \* Byzantine validator proposes low-quality block
           newBlock == [proposer |-> v, block |-> "malicious", quality |-> quality]
       IN proposals' = proposals \cup {newBlock}
          /\ UNCHANGED <<round, stake, reputation, contributions, blocks, votes, decided, messages>>

\* Honest validator votes for best valid block
VoteForBlock(v) ==
    /\ IsHonest(v)
    /\ proposals # {}
    /\ ~decided
    /\ \E p \in proposals :
        /\ p.quality >= MinQuality
        /\ votes' = [votes EXCEPT ![v] = [block |-> p.block, round |-> round]]
        /\ UNCHANGED <<round, stake, reputation, contributions, blocks, proposals, decided, messages>>

\* Byzantine validator votes (may vote for invalid blocks)
ByzantineVote(v) ==
    /\ ~IsHonest(v)
    /\ proposals # {}
    /\ ~decided
    /\ \E p \in proposals :
        votes' = [votes EXCEPT ![v] = [block |-> p.block, round |-> round]]
        /\ UNCHANGED <<round, stake, reputation, contributions, blocks, proposals, decided, messages>>

\* Finalize block if it has supermajority
FinalizeBlock ==
    /\ ~decided
    /\ \E p \in proposals :
        /\ HasSuperMajority(p.block)
        /\ blocks' = Append(blocks, [proposer |-> p.proposer, quality |-> p.quality, round |-> round])
        /\ decided' = TRUE
        /\ UNCHANGED <<round, stake, reputation, contributions, proposals, votes, messages>>

\* Move to next round if no decision
NextRound ==
    /\ ~decided
    /\ round < MaxRounds
    /\ ~(\E p \in proposals : HasSuperMajority(p.block))
    /\ round' = round + 1
    /\ proposals' = {}
    /\ votes' = [v \in Validators |-> [block |-> "none", round |-> round + 1]]
    /\ UNCHANGED <<stake, reputation, contributions, blocks, decided, messages>>

\* Update reputation based on contributions
UpdateReputation(v) ==
    /\ decided
    /\ LET lastBlock == blocks[Len(blocks)]
           adjustment == IF v = lastBlock.proposer 
                         THEN Min(100, reputation[v] + 10)
                         ELSE Max(0, reputation[v] - 1)
       IN reputation' = [reputation EXCEPT ![v] = adjustment]
          /\ UNCHANGED <<round, stake, contributions, blocks, proposals, votes, decided, messages>>

\* Slash malicious validator
SlashValidator(v) ==
    /\ ~IsHonest(v)
    /\ \E p \in proposals :
        /\ p.proposer = v
        /\ p.quality < MinQuality
        /\ stake' = [stake EXCEPT ![v] = stake[v] \div 2]  \* 50% slash
        /\ reputation' = [reputation EXCEPT ![v] = 0]      \* Reset reputation
        /\ UNCHANGED <<round, contributions, blocks, proposals, votes, decided, messages>>

-------------------------------------------------------------------------------------
\* Next State Relation
-------------------------------------------------------------------------------------

Next ==
    \/ \E v \in Validators : ProposeBlock(v)
    \/ \E v \in Validators : ProposeMaliciousBlock(v)
    \/ \E v \in Validators : VoteForBlock(v)
    \/ \E v \in Validators : ByzantineVote(v)
    \/ FinalizeBlock
    \/ NextRound
    \/ \E v \in Validators : UpdateReputation(v)
    \/ \E v \in Validators : SlashValidator(v)

Spec == Init /\ [][Next]_vars /\ WF_vars(Next)

-------------------------------------------------------------------------------------
\* Safety Properties
-------------------------------------------------------------------------------------

\* No two different blocks are finalized in the same round
Consistency ==
    \A i, j \in 1..Len(blocks) :
        (blocks[i].round = blocks[j].round) => (i = j)

\* Only high-quality blocks are finalized
QualityThreshold ==
    \A i \in 1..Len(blocks) :
        blocks[i].quality >= MinQuality

\* Byzantine validators cannot control consensus
ByzantineSafety ==
    LET byzantineStake == Sum(ByzantineValidators, TotalStake)
        totalStake == NetworkStake
    IN byzantineStake < totalStake \div 3

\* Total stake is conserved (except for slashing)
StakeConservation ==
    LET totalStake == Sum(Validators, stake)
        initialStake == Sum(Validators, LAMBDA v: 550)  \* Midpoint of 100..1000
    IN totalStake <= initialStake

-------------------------------------------------------------------------------------
\* Liveness Properties
-------------------------------------------------------------------------------------

\* Eventually some block is decided (weak liveness)
EventualDecision ==
    <>(decided = TRUE)

\* Progress is made in rounds
RoundProgress ==
    []<>(round < MaxRounds => round' = round + 1 \/ decided' = TRUE)

\* Good validators eventually have their blocks accepted
QualityReward ==
    \A v \in Validators :
        IsHonest(v) /\ contributions[v] >= 80 =>
            <>(\E i \in 1..Len(blocks) : blocks[i].proposer = v)

-------------------------------------------------------------------------------------
\* Temporal Properties
-------------------------------------------------------------------------------------

\* Once decided, stays decided
DecisionFinality ==
    [](decided => []decided)

\* Reputation monotonically improves for consistent good actors
ReputationGrowth ==
    \A v \in Validators :
        IsHonest(v) /\ contributions[v] >= 70 =>
            [](reputation[v] <= reputation'[v] \/ reputation'[v] = reputation[v] - 1)

-------------------------------------------------------------------------------------
\* Invariants to Check
-------------------------------------------------------------------------------------

SafetyInvariant ==
    /\ TypeOK
    /\ Consistency
    /\ QualityThreshold
    /\ ByzantineSafety
    /\ StakeConservation

LivenessProperty ==
    /\ EventualDecision
    /\ RoundProgress
    /\ DecisionFinality

-------------------------------------------------------------------------------------
\* Model Checking Configuration
-------------------------------------------------------------------------------------

\* For model checking with TLC, use small constants:
\* Validators = {v1, v2, v3, v4}
\* ByzantineValidators = {v4}
\* MaxRounds = 10
\* MinQuality = 30

================================================================================