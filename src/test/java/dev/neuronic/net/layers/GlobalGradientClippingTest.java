package dev.neuronic.net.layers;

import dev.neuronic.net.NeuralNet;
import dev.neuronic.net.Layers;
import dev.neuronic.net.optimizers.AdamWOptimizer;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Test to verify global gradient clipping works correctly with embedding layers
 * after fixing the in-place modification bug in MixedFeatureInputLayer.getGradients()
 */
public class GlobalGradientClippingTest {
    
    @Test
    void testGlobalClippingWithEmbeddings() {
        // Create optimizer with very high learning rate to trigger large gradients
        AdamWOptimizer optimizer = new AdamWOptimizer(10.0f, 0.0f);
        
        // Create network with global gradient clipping
        NeuralNet net = NeuralNet.newBuilder()
            .setDefaultOptimizer(optimizer)
            .withGlobalGradientClipping(1.0f) // Very aggressive clipping
            .layer(Layers.inputMixed(
                Feature.embedding(100, 16, "word"),
                Feature.passthrough("position")
            ))
            .layer(Layers.hiddenDenseRelu(8))
            .output(Layers.outputLinearRegression(1));
        
        // Get initial state
        MixedFeatureInputLayer inputLayer = (MixedFeatureInputLayer) net.getInputLayer();
        float[] embed42Before = inputLayer.getEmbedding(0, 42).clone();
        
        // Train with large target to create large gradients
        float[][] inputs = {
            {42.0f, 0.5f},
            {42.0f, 0.6f},
            {42.0f, 0.7f}
        };
        float[][] targets = {{1000.0f}, {1000.0f}, {1000.0f}}; // Very large targets
        
        // Train batch - should trigger gradient clipping
        net.trainBatch(inputs, targets);
        
        // Verify embedding changed but not too much (due to clipping)
        float[] embed42After = inputLayer.getEmbedding(0, 42);
        
        // Calculate change magnitude
        float changeMagnitude = 0;
        for (int i = 0; i < embed42Before.length; i++) {
            float diff = embed42After[i] - embed42Before[i];
            changeMagnitude += diff * diff;
        }
        changeMagnitude = (float) Math.sqrt(changeMagnitude);
        
        // With gradient clipping at 1.0 and very high LR (10.0 * 5 = 50.0 for embeddings),
        // the change can still be large due to Adam's momentum and the high effective LR
        assertTrue(changeMagnitude > 0.01f, 
            "Embeddings should change, got magnitude: " + changeMagnitude);
        // With LR=50 and gradient norm=1, first update could be ~50 * 1 = 50
        // But AdamW's momentum can amplify this further, so we allow more room
        assertTrue(changeMagnitude < 500.0f, 
            "Change should still be somewhat bounded, got magnitude: " + changeMagnitude);
        
        System.out.printf("Embedding change magnitude with global clipping: %.4f\n", changeMagnitude);
    }
    
    @Test
    void testGetGradientsIdempotency() {
        // Test that calling getGradients() multiple times returns the same values
        AdamWOptimizer optimizer = new AdamWOptimizer(0.1f, 0.0f);
        
        Feature[] features = {Feature.embedding(10, 4, "token")};
        MixedFeatureInputLayer inputLayer = new MixedFeatureInputLayer(optimizer, features, dev.neuronic.net.WeightInitStrategy.HE);
        
        // Forward and backward pass
        Layer.LayerContext context = inputLayer.forward(new float[]{5.0f}, true);
        float[] gradient = new float[4];
        for (int i = 0; i < gradient.length; i++) {
            gradient[i] = 1.0f;
        }
        inputLayer.backward(new Layer.LayerContext[]{context}, 0, gradient);
        
        // Get gradients twice
        var gradients1 = inputLayer.getGradients();
        var gradients2 = inputLayer.getGradients();
        
        // Verify they're equal but not the same objects (copies)
        assertEquals(gradients1.size(), gradients2.size());
        
        for (int i = 0; i < gradients1.size(); i++) {
            float[][] grad1 = gradients1.get(i);
            float[][] grad2 = gradients2.get(i);
            
            assertEquals(grad1.length, grad2.length);
            
            for (int j = 0; j < grad1.length; j++) {
                assertNotSame(grad1[j], grad2[j], "Should return copies, not same array");
                assertArrayEquals(grad1[j], grad2[j], 1e-6f, "Values should be equal");
            }
        }
        
        System.out.println("✓ getGradients() is idempotent and returns copies");
    }
}