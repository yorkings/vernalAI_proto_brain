import torch

class ProtoNeuron:
    def __init__(self,input_size,learning_rate:float=0.01,decay:float=0.9,init_scale:float=0.1):
        self.input_size=input_size
        self.learning_rate=learning_rate

        self.weights=(torch.randn(input_size)*2 -1)*init_scale
        self.bias=(torch.randn(1)*2 -1 )*init_scale

        self.trace=torch.tensor([float(0.0)])
        self.activation=torch.tanh
        self.gamma=decay

        self.last_input=None
        self.last_activation=None

    def forward(self,x):
        """
        Calculate output: z = Σ(w[i]*x[i]) + b, a = φ(z) 
        trace=γ⋅trace+a
        """
        self.last_input=x.clone()
        z=torch.dot(self.weights,x)+self.bias
        a=self.activation(z)
        self.trace=self.gamma*self.trace + a
        self.last_activation=a
        return a;

    def learn(self,target):
        """"
          Δw[i] = α * a * (t - a) * x[i]
          Δb   = α * (t - a)
        """  
        a=self.last_activation
        if isinstance(target,(int,float)):
            target=torch.tensor([float(target)])

        error=target - a

        delta_weight=self.learning_rate * a*error *self.last_input
        delta_bias=self.learning_rate*error
        self.weights+=delta_weight
        self.bias+=delta_bias


if __name__=="__main__":
    neuron=ProtoNeuron(input_size=4)
    test_input = torch.tensor([1.0, 0.5, -0.2,0.74])
    print('==INITIAL STATE==')
    print(f"Weights: {neuron.weights}")
    print(f"Bias: {neuron.bias}")
    print(f"Trace: {neuron.trace}")

    print("\n ==Forward pass==")
    output_1=neuron.forward(test_input)
    print(f"Input: {test_input}")
    print(f"Output: {output_1.item():.4f}")

    # Learn from reward
    print("\n=== LEARNING ===")
    print(f"Target (desired output): 0.8")
    print(f"Error: {0.8 - output_1.item():.4f}")
    neuron.learn(target=0.8)
    
    print(f"\nUpdated weights: {neuron.weights}")
    print(f"Updated bias: {neuron.bias}")
    print("\n=== FORWARD PASS 2 (same input) ===")
    output_2 = neuron.forward(test_input)
    print(f"Output: {output_2.item():.4f}")
    print(f"Change: {output_2.item() - output_1.item():.4f}")

    if abs(output_2.item() - 0.8) < abs(output_1.item() - 0.8):
        print("✅ SUCCESS:Neuron moved toward target!")
    else:
        print("❌ Neuron didn't learn correctly")
    # Test trace memory
    print("\n=== TEMPORAL MEMORY TEST ===")
    print(f"Trace after 2 activations: {neuron.trace.item():.4f}")
    print("\n" + "=" * 50)
    print("🎯 PROTO-NEURON COMPLETE!")












