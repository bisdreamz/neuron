package dev.neuronic.net.layers;

import dev.neuronic.net.optimizers.AdamWOptimizer;
import dev.neuronic.net.WeightInitStrategy;
import dev.neuronic.net.math.FastRandom;
import org.junit.jupiter.api.Test;
import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Test to verify that embedding updates are properly isolated.
 * This test specifically checks that when training on one embedding,
 * other embeddings remain unchanged.
 */
public class EmbeddingUpdateIsolationTest {
    
    @Test
    public void testEmbeddingUpdateIsolation() {
        // Create a simple embedding layer
        AdamWOptimizer optimizer = new AdamWOptimizer(0.1f, 0.0f);
        Feature[] features = {
            Feature.embedding(3, 4, "item") // 3 items, 4-dim embeddings
        };
        
        FastRandom random = new FastRandom(12345);
        MixedFeatureInputLayer layer = new MixedFeatureInputLayer(optimizer, features, WeightInitStrategy.HE, random);
        
        // Get initial embeddings
        float[] embed0_initial = layer.getEmbedding(0, 0).clone();
        float[] embed1_initial = layer.getEmbedding(0, 1).clone();
        float[] embed2_initial = layer.getEmbedding(0, 2).clone();
        
        System.out.println("Initial embeddings:");
        System.out.println("Item 0: " + Arrays.toString(embed0_initial));
        System.out.println("Item 1: " + Arrays.toString(embed1_initial));
        System.out.println("Item 2: " + Arrays.toString(embed2_initial));
        
        // Train ONLY on item 0
        Layer.LayerContext context = layer.forward(new float[]{0.0f}, true);
        float[] gradient = new float[4];
        Arrays.fill(gradient, 1.0f); // Large gradient to see clear change
        
        layer.backward(new Layer.LayerContext[]{context}, 0, gradient);
        layer.applyGradients(null, null);
        
        // Get updated embeddings
        float[] embed0_after = layer.getEmbedding(0, 0);
        float[] embed1_after = layer.getEmbedding(0, 1);
        float[] embed2_after = layer.getEmbedding(0, 2);
        
        System.out.println("\nAfter training on item 0:");
        System.out.println("Item 0: " + Arrays.toString(embed0_after) + " (should change)");
        System.out.println("Item 1: " + Arrays.toString(embed1_after) + " (should NOT change)");
        System.out.println("Item 2: " + Arrays.toString(embed2_after) + " (should NOT change)");
        
        // Verify item 0 changed
        boolean item0_changed = false;
        for (int i = 0; i < 4; i++) {
            if (Math.abs(embed0_initial[i] - embed0_after[i]) > 1e-6) {
                item0_changed = true;
                break;
            }
        }
        assertTrue(item0_changed, "Item 0 embedding should have changed after training");
        
        // Verify items 1 and 2 did NOT change
        assertArrayEquals(embed1_initial, embed1_after, 1e-6f, 
            "Item 1 embedding should NOT change when training on item 0");
        assertArrayEquals(embed2_initial, embed2_after, 1e-6f, 
            "Item 2 embedding should NOT change when training on item 0");
        
        // Calculate change magnitude for item 0
        float change_magnitude = 0;
        for (int i = 0; i < 4; i++) {
            float diff = embed0_after[i] - embed0_initial[i];
            change_magnitude += diff * diff;
        }
        change_magnitude = (float) Math.sqrt(change_magnitude);
        System.out.printf("\nItem 0 change magnitude: %.4f\n", change_magnitude);
        
        assertTrue(change_magnitude > 0.01f, "Item 0 should have significant change");
    }
}