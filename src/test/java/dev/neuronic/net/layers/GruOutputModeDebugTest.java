package dev.neuronic.net.layers;

import dev.neuronic.net.NeuralNet;
import dev.neuronic.net.Layers;
import dev.neuronic.net.optimizers.SgdOptimizer;
import org.junit.jupiter.api.Test;

/**
 * Debug test to understand GRU output size calculation.
 */
public class GruOutputModeDebugTest {
    
    @Test
    public void debugGruSizeCalculation() {
        System.out.println("\n=== GRU Size Calculation Debug ===");
        
        int sequenceLength = 10;
        int embeddingDim = 16;
        int hiddenSize = 32;
        
        // Create specs and check their output sizes
        Layer.Spec embeddingSpec = Layers.inputEmbedding(1000, embeddingDim);
        System.out.println("Embedding spec output size (static): " + embeddingSpec.getOutputSize());
        System.out.println("Embedding spec output size (with input " + sequenceLength + "): " + 
                          embeddingSpec.getOutputSize(sequenceLength));
        
        // The embedding layer will output seqLen × embeddingDim = 10 × 16 = 160
        int expectedEmbeddingOutput = sequenceLength * embeddingDim;
        System.out.println("Expected embedding output: " + expectedEmbeddingOutput);
        
        // Create GRU spec with hint
        SgdOptimizer optimizer = new SgdOptimizer(0.01f);
        Layer.Spec gruSpec = Layers.hiddenGruAll(hiddenSize, optimizer, embeddingDim);
        System.out.println("\nGRU spec output size (static): " + gruSpec.getOutputSize());
        System.out.println("GRU spec output size (with input " + expectedEmbeddingOutput + "): " + 
                          gruSpec.getOutputSize(expectedEmbeddingOutput));
        
        // Expected GRU output: seqLen × hiddenSize = 10 × 32 = 320
        int expectedGruOutput = sequenceLength * hiddenSize;
        System.out.println("Expected GRU output: " + expectedGruOutput);
        
        // Now let's trace through actual layer creation
        System.out.println("\n=== Building Network ===");
        try {
            NeuralNet model = NeuralNet.newBuilder()
                    .input(sequenceLength)
                    .setDefaultOptimizer(optimizer)
                    .layer(embeddingSpec)
                    .layer(gruSpec)
                    .layer(Layers.hiddenDenseRelu(64))
                    .output(Layers.outputSoftmaxCrossEntropy(10));
            
            System.out.println("Network built successfully!");
            
            // Test forward pass
            float[] input = new float[sequenceLength];
            for (int i = 0; i < sequenceLength; i++) {
                input[i] = i % 100;
            }
            
            float[] output = model.predict(input);
            System.out.println("Forward pass successful! Output size: " + output.length);
            
        } catch (Exception e) {
            System.out.println("Failed to build network: " + e.getMessage());
            e.printStackTrace();
        }
    }
}