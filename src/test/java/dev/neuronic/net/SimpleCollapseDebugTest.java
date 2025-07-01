package dev.neuronic.net;

import dev.neuronic.net.layers.Feature;
import dev.neuronic.net.optimizers.AdamWOptimizer;
import dev.neuronic.net.optimizers.SgdOptimizer;
import dev.neuronic.net.simple.SimpleNet;
import dev.neuronic.net.simple.SimpleNetFloat;
import org.junit.jupiter.api.Test;

import java.util.*;

/**
 * Simple test to isolate collapse bug.
 * TODO: REMOVE THIS FILE AFTER DEBUGGING
 */
public class SimpleCollapseDebugTest {
    
    @Test
    public void testMinimalCollapseCase() {
        System.out.println("=== MINIMAL COLLAPSE TEST ===\n");
        
        // Ultra simple: just one embedding feature
        Feature[] features = {
            Feature.embedding(10, 4, "item")  // 10 items, 4-dim embeddings
        };
        
        // Try both optimizers
        testWithOptimizer(new SgdOptimizer(0.1f), "SGD", features);
        testWithOptimizer(new AdamWOptimizer(0.01f, 0.001f), "AdamW", features);
    }
    
    private void testWithOptimizer(dev.neuronic.net.optimizers.Optimizer optimizer, String name, Feature[] features) {
        System.out.println("--- Testing with " + name + " ---");
        
        NeuralNet net = NeuralNet.newBuilder()
            .input(features.length)
            .setDefaultOptimizer(optimizer)
            .layer(Layers.inputMixed(features))
            .layer(Layers.hiddenDenseRelu(32))
            .output(Layers.outputLinearRegression(1));
            
        SimpleNetFloat model = SimpleNet.ofFloatRegression(net);
        
        // Train items to different targets
        Map<Integer, Float> targets = Map.of(
            0, 1.0f,
            1, 0.5f,
            2, -0.5f,
            3, -1.0f
        );
        
        // Train each item multiple times
        for (int epoch = 0; epoch < 50; epoch++) {
            for (Map.Entry<Integer, Float> entry : targets.entrySet()) {
                Map<String, Object> input = Map.of("item", entry.getKey());
                model.train(input, entry.getValue());
            }
        }
        
        // Check predictions
        System.out.println("Predictions after training:");
        Set<String> uniquePreds = new HashSet<>();
        for (int i = 0; i < 4; i++) {
            Map<String, Object> input = Map.of("item", i);
            float pred = model.predictFloat(input);
            String predStr = String.format("%.3f", pred);
            uniquePreds.add(predStr);
            System.out.printf("  Item %d: %.3f (target: %.1f)\\n", i, pred, targets.get(i));
        }
        
        System.out.printf("Unique predictions: %d\\n", uniquePreds.size());
        if (uniquePreds.size() == 1) {
            System.out.println("⚠️  COLLAPSED!");
        } else {
            System.out.println("✓ Diverse predictions");
        }
        System.out.println();
    }
}