import torch
from Phase_one_neuron_layer import ProtoNeuron
"""lr=learning-rate """
class ProtoLayer:
    def __init__(self,input_size:int,lr:float=0.01,num_neurons:int=5,feedback_strength:float=0.1):
        self.input_size=input_size
        self.lr=lr
        self.num_neurons=num_neurons
        self.feedback_strength=feedback_strength

        self.neurons=[]
        self.last_outputs=None

        for _ in range(num_neurons):
            neuron=ProtoNeuron(input_size,learning_rate=lr)
            self.neurons.append(neuron)

    def forward(self,x):
        """
        1. output[i] = neuron[i].forward(x)

        2. feedback = β * mean(output)
           output[i] += feedback

        3. stabilization:
           out = out / (1 + |out|)
        """
        if len(x) != self.input_size:
          raise ValueError(f"Input size {len(x)} doesn't match {self.input_size}")
        
        outputs=torch.zeros(self.num_neurons)
        for i,neuron in enumerate(self.neurons):
            outputs[i]=neuron.forward(x)

        mean_outputs= torch.mean(outputs)
        feedback=self.feedback_strength *mean_outputs
        outputs+=feedback
        #stabilize
        outputs = outputs / (1 +torch.abs(outputs))
        #make it non linear
        outputs = torch.tanh(outputs)
        self.last_outputs=outputs
        return outputs
    
    def learn(self,target):
        for neuron in self.neurons:
           neuron.learn(target)



if __name__ == "__main__":
    print("--- Starting ProtoLayer Test Script ---")
    
    INPUT_DIM = 4
    NUM_NEURONS = 5
    LEARNING_RATE = 0.01
    FEEDBACK_STR = 0.1

    # Initialize the ProtoLayer
    try:
        proto_layer = ProtoLayer(
            input_size=INPUT_DIM,
            lr=LEARNING_RATE,
            num_neurons=NUM_NEURONS,
            feedback_strength=FEEDBACK_STR
        )
        print(f"Successfully initialized ProtoLayer with {NUM_NEURONS} neurons.")
    except Exception as e:
        print(f"Initialization failed: {e}")
        exit()

    # --- Test 1: Forward Pass with Valid Input ---
    print("\n--- Test 1: Forward Pass ---")
    # Create a dummy input vector (e.g., a single sample from a dataset)
    input_vector = torch.randn(INPUT_DIM) 
    print(f"Input vector: {input_vector.numpy()}")

    try:
        output_result = proto_layer.forward(input_vector)
        print(f"Output tensor shape: {output_result.shape}")
        print(f"Output values (tanh applied): {output_result.detach().numpy()}")
        
        # Basic assertions
        assert output_result.shape[0] == NUM_NEURONS
        assert torch.all((output_result >= -1) & (output_result <= 1)) # Check tanh bounds
        print("Forward pass successful and within expected bounds (-1, 1).")

    except ValueError as e:
        print(f"Forward pass failed with expected ValueError for shape mismatch: {e}")
    except Exception as e:
        print(f"Forward pass failed unexpectedly: {e}")

    
    # --- Test 2: Forward Pass with Invalid Input Size (Error Handling) ---
    print("\n--- Test 2: Invalid Input Handling ---")
    invalid_input = torch.randn(INPUT_DIM + 1) # Too many dimensions
    try:
        proto_layer.forward(invalid_input)
        print("Error: Forward pass should have raised a ValueError but did not.")
    except ValueError as e:
        print(f"Successfully caught expected error: {e}")

    # --- Test 3: Learning/Update Step (requires `ProtoNeuron.learn` implementation) ---
    print("\n--- Test 3: Learning Step ---")
    # Define a dummy target value (the exact nature depends on how `ProtoNeuron.learn` works internally)
    dummy_target = 0.5 
    print(f"Applying learn step with target: {dummy_target}")
    try:
        # Note: The `learn` function as written calls `neuron.learn(target)` 
        # for every neuron, which might be a strange learning rule depending on context.
        # This test just ensures it runs without crashing.
        proto_layer.learn(dummy_target)
        print("Learning step executed without errors.")
    except Exception as e:
        print(f"Learning step failed: {e}")
        
    print("\n--- Testing Complete ---")

        



