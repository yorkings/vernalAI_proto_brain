class ChainBuilder:
    def __init__(self, forward_sim, actions, K=4):
        self.forward_sim = forward_sim
        self.actions = actions
        self.K = K
        self.chain_memory = {}  # fingerprint -> (chain, success_count, failure_count)
        self.memory_capacity = 200
        
    def build_chain(self, state, goal, momentum_system):
        print(f"[CHAIN DEBUG] State: {state[:3]}...")  
        print(f"[CHAIN DEBUG] Goal: {goal[:3]}...")
        chain = []
        predicted_states = []
        s = state.clone()
        
        distance_weight = 2.0  # ADD THIS
        momentum_weight = 0.5  # ADD THIS
        
        for k in range(self.K):
            print(f"\n  Step {k}: Current state norm: {s.norm().item():.3f}")
            best_action = None
            best_score = -float('inf')
            
            for i, action_vec in enumerate(self.actions):
                predicted_next = self.forward_sim.imagine_step(s, action_vec)
                distance = (predicted_next - goal).norm().item()
                momentum_bonus = momentum_system.get_momentum_bonus(i)
                
                # Distance should be positive, we want to minimize it
                distance_penalty = distance_weight * distance
                score = -distance_penalty + momentum_weight * momentum_bonus
                                   
                print(f"    Action {i}: distance={distance:.3f}, "
                    f"momentum={momentum_bonus:.3f}, score={score:.3f}")
                
                if score > best_score:
                    best_score = score
                    best_action = (i, action_vec)
            
            if best_action:
                action_idx, action_vec = best_action
                chain.append(action_idx)
                s = self.forward_sim.imagine_step(s, action_vec)
                predicted_states.append(s.clone())
                print(f"  Chosen action: {action_idx}, new state norm: {s.norm().item():.3f}")
        
        print(f"[CHAIN DEBUG] Final chain: {chain}")
        return chain, predicted_states