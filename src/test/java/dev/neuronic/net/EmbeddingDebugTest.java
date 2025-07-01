package dev.neuronic.net;

import dev.neuronic.net.layers.Feature;
import dev.neuronic.net.optimizers.SgdOptimizer;
import dev.neuronic.net.simple.SimpleNet;
import dev.neuronic.net.simple.SimpleNetFloat;
import org.junit.jupiter.api.Test;

import java.util.*;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Debug test to find where embeddings are incorrectly updating.
 * TODO: REMOVE THIS FILE AFTER DEBUGGING
 */
public class EmbeddingDebugTest {
    
    @Test
    public void debugEmbeddingUpdates() {
        System.out.println("=== DEBUG: EMBEDDING UPDATE TRACKING ===\n");
        
        // Minimal case: 3 embeddings
        Feature[] features = {
            Feature.embedding(3, 2, "item")  // 3 items, 2-dim embeddings
        };
        
        SgdOptimizer optimizer = new SgdOptimizer(1.0f); // LR=1 for clear updates
        
        NeuralNet net = NeuralNet.newBuilder()
            .input(features.length)
            .setDefaultOptimizer(optimizer)
            .layer(Layers.inputMixed(features))
            .output(Layers.outputLinearRegression(1));
            
        SimpleNetFloat model = SimpleNet.ofFloatRegression(net);
        
        // Get initial embeddings
        System.out.println("Initial embeddings:");
        for (int i = 0; i < 3; i++) {
            Map<String, Object> input = new HashMap<>();
            input.put("item", i);
            float pred = model.predictFloat(input);
            System.out.printf("  Item %d: prediction = %.3f\n", i, pred);
        }
        
        // Train ONLY item 0
        System.out.println("\nTraining item 0 -> target 10.0");
        System.out.println("SGD with LR=1.0, so weight change = gradient");
        
        Map<String, Object> input0 = new HashMap<>();
        input0.put("item", 0);
        
        // Enable debug mode
        System.setProperty("neuronic.debug.embeddings", "true");
        
        model.train(input0, 10.0f);
        
        // Disable debug mode
        System.clearProperty("neuronic.debug.embeddings");
        
        // Check all embeddings after
        System.out.println("\nAfter training:");
        for (int i = 0; i < 3; i++) {
            Map<String, Object> input = new HashMap<>();
            input.put("item", i);
            float pred = model.predictFloat(input);
            System.out.printf("  Item %d: prediction = %.3f\n", i, pred);
        }
    }
}