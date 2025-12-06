# 🧠 Cortical Intelligence Engine (CorteX)

A biologically-inspired hierarchical learning architecture mimicking mammalian neocortex organization, featuring cortical columns, dopamine modulation, attractor dynamics, and global workspace integration.

## 🌟 Project Vision
Build a cognitive architecture that combines neuroscience principles with machine learning to create systems capable of continuous learning, temporal binding, and compositional reasoning without catastrophic forgetting.

## 📁 Project Structure

### Phase 1: Foundation Layer (Completed)
**Location:** `phase_one_life_loop.py`
**Goal:** Basic neural circuits with local learning in simple environment

#### Components:
- **`ProtoNeuron`** - Basic neuron with Hebbian learning
  - Local learning rule: Δw = α * a * (t - a) * x
  - Temporal trace memory (decay: γ)
  - Tanh activation with trace-based memory

- **`ProtoLayer`** - Column-like neural layer
  - Multiple neurons sharing same input
  - Feedback connections between neurons
  - Stabilization: output / (1 + |output|)
  - Lateral inhibition/cooperation via feedback strength

- **`World1D`** - Simple 1D environment
  - Position-based navigation task
  - Goal at position 0.0
  - Reward = -|distance_to_goal|
  - Action space: {-1, 0, +1}

- **`SimpleAgent`** - Basic policy
  - Maps neuron outputs to actions
  - Action = sign(first_neuron_output)

- **`run_life_loop()`** - Main training
  - Sense → Process → Act → Reward → Learn cycle
  - Logs position, reward, actions for analysis

#### Key Achievement:
Demonstrated local Hebbian learning enabling simple navigation in 1D world.

---

### Phase 2: Hierarchical Cortical System (In Progress)

#### Phase 2.1: Dopamine-Modulated Neurons
**Component:** `ProtoNeuronMod` (in `phase_two_modulated_neuron.py`)
**Innovation:** Three-factor learning rule with eligibility traces

**Features:**
- **Eligibility traces** - Temporal memory of pre-post activity
- **Dopamine modulation** - δ * eligibility * plasticity_gate
- **Local + DA blended learning** - mix_local parameter controls balance
- **Plasticity gates** - Control when learning occurs
- **Trace decay** (λ) for temporal credit assignment

**Learning Rule:**
Δw = mix_local * [α_local * a * error * x] +
(1-mix_local) * [α_da * δ * eligibility * gate]



#### Phase 2.2: Cortical Columns
**Component:** `ProtoColumn` (in `phase_two_column.py`)
**Concept:** Microcircuit organization mimicking cortical columns

**Features:**
- **Columnar organization** - Multiple neurons as functional unit
- **Lateral inhibition** - hi′ = hi - λ(hi - h̄)
- **Cooperative excitation** - c = h + η⋅mean(a)
- **Winner-biased normalization** - Out = excited / (|max| + ε)
- **Column trace** - Slow state memory: T ← γT + ō
- **Trace-modulated targets** - target + 0.05*trace

**Dynamics:**
1. Compute neuron activations
2. Apply lateral inhibition (sparsification)
3. Add cooperative excitation (feature binding)
4. Winner-take-all normalization
5. Update column trace memory

#### Phase 2.3: Cortical Blocks
**Component:** `CorticalBlock` (in `phase_two_cortex.py`)
**Concept:** Recurrent network of columns forming cortical areas

**Features:**
- **Recurrent connections** - z = (1-ρ)z + ρ(Wo + Bx)
- **Inter-column weights** - Learn via Oja's rule
- **Temporal dynamics** - Slow recurrent state (ρ leak rate)
- **Block eligibility** - Matrix trace for dopamine modulation
- **Input projection** - Optional B matrix for sensory input
- **Reduce function** - Column vector → scalar (mean, max, norm)

**Learning Rule (Oja + DA):**
ΔW = η * (ooᵀ - diag(o²)W) + η * δ * eligibility - decay*W



#### Phase 2.4: Attractor Cortex
**Component:** `AttractorCortex` (in `phase_two_attractor.py`)
**Concept:** Stable attractor dynamics for working memory

**Features:**
- **Settling dynamics** - Iterative convergence to stable state
- **Energy minimization** - E(o) = -½(oᵀWo) + λ∑o²
- **Temporal prediction** - W_temp for sequence learning
- **Homeostasis** - Firing rate regulation
- **Synaptic scaling** - Maintain weight norms
- **Sparse coding** - k-WTA (hard/soft) for efficient representations
- **Memory replay** - Offline consolidation during rest

**Key Mechanisms:**
1. **Settling loop** (up to N steps to convergence)
2. **Energy computation** for stability
3. **Sparsity enforcement** (k-winner take-all)
4. **Homeostatic regulation** (activation_target maintenance)
5. **Temporal learning** (predict next state)

#### Phase 2.5: Global Workspace
**Component:** `GlobalWorkspace` (in `phase_two_GWI.py`)
**Concept:** Consciousness-like global ignition and broadcasting

**Features:**
- **Ignition detection** - Threshold-based global events
- **Importance scoring** - 0.5*salience + 0.3*surprise + 0.2*reward
- **Broadcast vector** - Fused representation for global access
- **Sync mask** - Which columns participate in ignition
- **Fusion network** - Learns to combine column outputs

**Ignition Process:**
1. Compute column importance scores
2. Apply threshold → ignition_mask
3. Weighted fusion of active columns
4. Broadcast to all columns
5. Synchronize processing

#### Phase 2.6: Reward & Valuation
**Component:** `RewardModule` (in `phase_two_valuation.py`)
**Concept:** Dopamine-like reward prediction error system

**Features:**
- **Value estimation** - V = expected future reward
- **RPE computation** - δ = r - V
- **TD-like learning** - V ← V + α*δ
- **Baseline reset** - V = 0.0 on environment reset

**Equation:**
δ(t) = r(t) - V(t) # Reward Prediction Error
V(t+1) = V(t) + α*δ(t) # Value update


### Phase 3: Planned Extensions (Future)

#### Phase 3.1: Multi-Modal Integration
- Visual, auditory, somatosensory streams
- Cross-modal association learning
- Unified representations across modalities

#### Phase 3.2: Hierarchical Planning
- Prefrontal cortex-like planning
- Goal-directed behavior generation
- Temporal abstraction (skills, subgoals)

#### Phase 3.3: Motor Control
- Cerebellum-like fine motor control
- Basal ganglia-like action selection
- Proprioceptive feedback integration

#### Phase 3.4: Memory Systems
- Hippocampal episodic memory
- Neocortical semantic memory consolidation
- Sleep-like memory replay optimization

---

## 🧬 Key Biological Principles Implemented

### 1. **Columnar Organization**
- Cortical column microcircuits
- Minicolumn/hypercolumn hierarchy
- Vertical information flow

### 2. **Neuromodulation**
- Dopamine-based three-factor learning
- Eligibility traces for temporal credit
- Plasticity gates for when learning occurs

### 3. **Temporal Dynamics**
- Theta-gamma phase coding
- Attractor states for working memory
- Eligibility traces for TD learning

### 4. **Homeostatic Regulation**
- Synaptic scaling (constant weight norms)
- Firing rate homeostasis
- Energy minimization in attractors

### 5. **Sparse Distributed Coding**
- k-Winner-Take-All competition
- Hard/soft sparsity options
- Efficient information representation

### 6. **Global Coordination**
- Threshold-based global ignition
- Broadcast-and-synchronize mechanism
- Consciousness-like access unity

---

## 🔬 Scientific Foundations

### Inspired By:
1. **Mountcastle's Cortical Column Theory**
2. **Dehaene's Global Neuronal Workspace**
3. **Sutton & Barto's Temporal Difference Learning**
4. **Buzsáki's Theta-Gamma Oscillations**
5. **Oja's Subspace Learning Rules**
6. **Hopfield's Attractor Networks**
7. **Turrigiano's Homeostatic Plasticity**

### Novel Integration:
- **First system** combining all above principles
- **Practical implementation** focus (not just simulation)
- **Scalable hierarchy** from neurons to cortical areas
- **Continuous learning** without catastrophic forgetting

---

## 🚀 Getting Started

### Prerequisites:
```bash
pip install torch numpy matplotlib