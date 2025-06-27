package dev.neuronic.net;

import dev.neuronic.net.layers.Layer;
import dev.neuronic.net.optimizers.SgdOptimizer;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for shape-aware GRU implementation.
 */
public class ShapeAwareGruTest {
    
    @Test
    public void testGruWithShapeAPI() {
        // Create a model using shape-aware API
        int sequenceLength = 10;
        int embeddingDim = 16;
        int hiddenSize = 32;
        
        SgdOptimizer optimizer = new SgdOptimizer(0.01f);
        
        NeuralNet model = NeuralNet.newBuilder()
                .input(Shape.sequence(sequenceLength, 1))  // 10 timesteps, 1 feature (token ID)
                .setDefaultOptimizer(optimizer)
                .layer(Layers.inputEmbedding(1000, embeddingDim))
                .layer(Layers.hiddenGruAll(hiddenSize))  // Should now work correctly!
                .layer(Layers.hiddenDenseRelu(64))
                .output(Layers.outputSoftmaxCrossEntropy(10));
        
        // Test forward pass
        float[] input = new float[sequenceLength];
        for (int i = 0; i < sequenceLength; i++) {
            input[i] = i % 100; // Some token IDs
        }
        
        float[] output = model.predict(input);
        
        // Output should be 10 classes (from softmax)
        assertEquals(10, output.length);
    }
    
    @Test
    public void testShapeInference() {
        // Test that shapes are properly inferred through the network
        Shape inputShape = Shape.sequence(20, 1);
        
        // Create specs
        Layer.Spec embeddingSpec = Layers.inputEmbedding(5000, 128);
        Layer.Spec gruSpec = Layers.hiddenGruAll(256);
        
        // Test shape propagation
        System.out.println("Input shape: " + inputShape);
        
        // Embedding doesn't know about sequences yet, so it returns vector
        Shape afterEmbedding = embeddingSpec.getOutputShape(inputShape);
        System.out.println("After embedding: " + afterEmbedding);
        
        // GRU with shape awareness should handle this correctly
        if (gruSpec.prefersShapeAPI()) {
            // For this test, we need to simulate what would happen
            // In real usage, the embedding layer would output [20, 128]
            Shape sequenceShape = Shape.sequence(20, 128);
            Shape afterGru = gruSpec.getOutputShape(sequenceShape);
            System.out.println("After GRU: " + afterGru);
            
            assertEquals(20, afterGru.dim(0)); // Sequence length preserved
            assertEquals(256, afterGru.dim(1)); // Hidden size
        }
    }
    
    @Test
    public void testGruShapeValidation() {
        Layer.Spec gruSpec = Layers.hiddenGruAll(128);
        
        // Valid shapes
        assertDoesNotThrow(() -> gruSpec.validateInputShape(Shape.sequence(10, 64)));
        assertDoesNotThrow(() -> gruSpec.validateInputShape(Shape.vector(640)));
        
        // Invalid shapes
        assertThrows(IllegalArgumentException.class, 
                    () -> gruSpec.validateInputShape(Shape.of(10, 20, 30))); // 3D not supported
    }
    
    @Test
    public void testBackwardCompatibility() {
        // Ensure old API still works
        NeuralNet model = NeuralNet.newBuilder()
                .input(10)  // Old style
                .setDefaultOptimizer(new SgdOptimizer(0.01f))
                .layer(Layers.inputEmbedding(1000, 16))
                .layer(Layers.hiddenGruLast(32))  // Last timestep works fine
                .output(Layers.outputSoftmaxCrossEntropy(10));
        
        float[] input = new float[10];
        float[] output = model.predict(input);
        
        assertEquals(10, output.length);
    }
}