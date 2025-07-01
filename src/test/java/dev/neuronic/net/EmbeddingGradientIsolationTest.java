package dev.neuronic.net;

import dev.neuronic.net.layers.Feature;
import dev.neuronic.net.layers.Layer;
import dev.neuronic.net.layers.Layer.LayerContext;
import dev.neuronic.net.layers.MixedFeatureInputLayer;
import dev.neuronic.net.optimizers.SgdOptimizer;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

public class EmbeddingGradientIsolationTest {
    
    @Test
    public void testEmbeddingGradientIsolation() {
        // Test that only the embedding being trained receives gradients
        SgdOptimizer optimizer = new SgdOptimizer(1.0f); // High learning rate to see changes clearly
        
        Feature[] features = {
            Feature.embedding(3, 4, "item") // 3 embeddings, 4-dimensional
        };
        
        MixedFeatureInputLayer layer = new MixedFeatureInputLayer(optimizer, features, WeightInitStrategy.HE);
        
        // Get initial embeddings
        float[] embedding0_before = layer.getEmbedding(0, 0).clone();
        float[] embedding1_before = layer.getEmbedding(0, 1).clone();
        float[] embedding2_before = layer.getEmbedding(0, 2).clone();
        
        System.out.println("Initial embeddings:");
        System.out.println("Embedding 0: " + java.util.Arrays.toString(embedding0_before));
        System.out.println("Embedding 1: " + java.util.Arrays.toString(embedding1_before));
        System.out.println("Embedding 2: " + java.util.Arrays.toString(embedding2_before));
        
        // Forward pass with embedding 0
        float[] input = {0.0f}; // Use embedding at index 0
        LayerContext context = layer.forward(input, true);
        
        // Create gradient for backward pass (all 1s to see clear changes)
        float[] gradient = new float[4];
        for (int i = 0; i < 4; i++) {
            gradient[i] = 1.0f;
        }
        
        // Backward pass
        LayerContext[] stack = {context};
        layer.backward(stack, 0, gradient);
        
        // Apply gradients
        layer.applyGradients(null, null);
        
        // Get embeddings after update
        float[] embedding0_after = layer.getEmbedding(0, 0);
        float[] embedding1_after = layer.getEmbedding(0, 1);
        float[] embedding2_after = layer.getEmbedding(0, 2);
        
        System.out.println("\nAfter training embedding 0:");
        System.out.println("Embedding 0: " + java.util.Arrays.toString(embedding0_after));
        System.out.println("Embedding 1: " + java.util.Arrays.toString(embedding1_after));
        System.out.println("Embedding 2: " + java.util.Arrays.toString(embedding2_after));
        
        // Check that only embedding 0 changed
        boolean embedding0_changed = false;
        for (int i = 0; i < 4; i++) {
            if (Math.abs(embedding0_after[i] - embedding0_before[i]) > 1e-6) {
                embedding0_changed = true;
                break;
            }
        }
        
        boolean embedding1_changed = false;
        for (int i = 0; i < 4; i++) {
            if (Math.abs(embedding1_after[i] - embedding1_before[i]) > 1e-6) {
                embedding1_changed = true;
                break;
            }
        }
        
        boolean embedding2_changed = false;
        for (int i = 0; i < 4; i++) {
            if (Math.abs(embedding2_after[i] - embedding2_before[i]) > 1e-6) {
                embedding2_changed = true;
                break;
            }
        }
        
        System.out.println("\nChanges detected:");
        System.out.println("Embedding 0 changed: " + embedding0_changed);
        System.out.println("Embedding 1 changed: " + embedding1_changed);
        System.out.println("Embedding 2 changed: " + embedding2_changed);
        
        // Assertions
        assertTrue(embedding0_changed, "Embedding 0 should have changed (it was trained)");
        assertFalse(embedding1_changed, "Embedding 1 should NOT have changed (it was not used)");
        assertFalse(embedding2_changed, "Embedding 2 should NOT have changed (it was not used)");
        
        // Verify the expected gradient update (SGD with lr=1.0, gradient=1.0)
        // Expected: embedding0_after[i] = embedding0_before[i] - 1.0 * 1.0
        for (int i = 0; i < 4; i++) {
            float expected = embedding0_before[i] - 1.0f; // SGD update
            assertEquals(expected, embedding0_after[i], 1e-5, 
                "Embedding 0 dimension " + i + " should be updated correctly");
        }
    }
    
    @Test 
    public void testMultipleBatchGradientIsolation() {
        // Test gradient isolation across multiple training steps
        SgdOptimizer optimizer = new SgdOptimizer(0.1f);
        
        Feature[] features = {
            Feature.embedding(5, 3, "item") // 5 embeddings, 3-dimensional
        };
        
        MixedFeatureInputLayer layer = new MixedFeatureInputLayer(optimizer, features, WeightInitStrategy.HE);
        
        // Train different embeddings in sequence
        for (int embIdx = 0; embIdx < 3; embIdx++) {
            // Record all embeddings before training
            float[][] allEmbeddings_before = new float[5][];
            for (int i = 0; i < 5; i++) {
                allEmbeddings_before[i] = layer.getEmbedding(0, i).clone();
            }
            
            // Train only embedding embIdx
            float[] input = {(float) embIdx};
            LayerContext context = layer.forward(input, true);
            
            float[] gradient = {1.0f, 1.0f, 1.0f}; // Gradient for 3-dimensional embedding
            LayerContext[] stack = {context};
            layer.backward(stack, 0, gradient);
            layer.applyGradients(null, null);
            
            // Check that ONLY the trained embedding changed
            for (int i = 0; i < 5; i++) {
                float[] after = layer.getEmbedding(0, i);
                boolean changed = false;
                
                for (int j = 0; j < 3; j++) {
                    if (Math.abs(after[j] - allEmbeddings_before[i][j]) > 1e-6) {
                        changed = true;
                        break;
                    }
                }
                
                if (i == embIdx) {
                    assertTrue(changed, "Embedding " + i + " should have changed (it was trained)");
                } else {
                    assertFalse(changed, "Embedding " + i + " should NOT have changed when training embedding " + embIdx);
                }
            }
        }
    }
    
    @Test
    public void testBatchAccumulationGradientIsolation() {
        // Test that gradients accumulate correctly within a batch but are isolated between embeddings
        SgdOptimizer optimizer = new SgdOptimizer(1.0f);
        
        Feature[] features = {
            Feature.embedding(3, 2, "item") // 3 embeddings, 2-dimensional
        };
        
        MixedFeatureInputLayer layer = new MixedFeatureInputLayer(optimizer, features, WeightInitStrategy.HE);
        
        // Get initial embeddings
        float[] embedding0_initial = layer.getEmbedding(0, 0).clone();
        float[] embedding1_initial = layer.getEmbedding(0, 1).clone();
        float[] embedding2_initial = layer.getEmbedding(0, 2).clone();
        
        // Simulate a batch with multiple samples using different embeddings
        // Sample 1: Use embedding 0
        float[] input1 = {0.0f};
        LayerContext context1 = layer.forward(input1, true);
        float[] gradient1 = {1.0f, 1.0f};
        layer.backward(new LayerContext[]{context1}, 0, gradient1);
        
        // Sample 2: Use embedding 1
        float[] input2 = {1.0f};
        LayerContext context2 = layer.forward(input2, true);
        float[] gradient2 = {2.0f, 2.0f};
        layer.backward(new LayerContext[]{context2}, 0, gradient2);
        
        // Sample 3: Use embedding 0 again (should accumulate)
        float[] input3 = {0.0f};
        LayerContext context3 = layer.forward(input3, true);
        float[] gradient3 = {3.0f, 3.0f};
        layer.backward(new LayerContext[]{context3}, 0, gradient3);
        
        // Apply accumulated gradients
        layer.applyGradients(null, null);
        
        // Get updated embeddings
        float[] embedding0_final = layer.getEmbedding(0, 0);
        float[] embedding1_final = layer.getEmbedding(0, 1);
        float[] embedding2_final = layer.getEmbedding(0, 2);
        
        // Verify updates
        // Embedding 0 should have accumulated gradients from samples 1 and 3
        // Total gradient: (1+3)/3 = 4/3 (averaged over batch size 3)
        float expectedGradient0 = 4.0f / 3.0f;
        assertEquals(embedding0_initial[0] - expectedGradient0, embedding0_final[0], 1e-5,
            "Embedding 0 dim 0 should be updated with accumulated gradient");
        assertEquals(embedding0_initial[1] - expectedGradient0, embedding0_final[1], 1e-5,
            "Embedding 0 dim 1 should be updated with accumulated gradient");
        
        // Embedding 1 should have gradient from sample 2 only
        // Total gradient: 2/3 (averaged over batch size 3)
        float expectedGradient1 = 2.0f / 3.0f;
        assertEquals(embedding1_initial[0] - expectedGradient1, embedding1_final[0], 1e-5,
            "Embedding 1 dim 0 should be updated with its gradient");
        assertEquals(embedding1_initial[1] - expectedGradient1, embedding1_final[1], 1e-5,
            "Embedding 1 dim 1 should be updated with its gradient");
        
        // Embedding 2 should not change
        assertEquals(embedding2_initial[0], embedding2_final[0], 1e-5,
            "Embedding 2 dim 0 should not change");
        assertEquals(embedding2_initial[1], embedding2_final[1], 1e-5,
            "Embedding 2 dim 1 should not change");
    }
}