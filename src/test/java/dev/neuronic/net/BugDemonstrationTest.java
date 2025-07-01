package dev.neuronic.net;

import dev.neuronic.net.layers.Feature;
import dev.neuronic.net.optimizers.SgdOptimizer;
import dev.neuronic.net.simple.SimpleNet;
import dev.neuronic.net.simple.SimpleNetFloat;
import org.junit.jupiter.api.Test;

import java.util.*;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Demonstrates why the embedding bug wasn't caught by existing tests.
 */
public class BugDemonstrationTest {
    
    @Test
    public void testWhyBugPassedTests() {
        System.out.println("=== WHY THE BUG PASSED TESTS ===\n");
        
        // Simple case: 3 embeddings
        Feature[] features = {
            Feature.embedding(3, 4, "item")
        };
        
        SgdOptimizer optimizer = new SgdOptimizer(0.1f);
        
        NeuralNet net = NeuralNet.newBuilder()
            .input(features.length)
            .setDefaultOptimizer(optimizer)
            .layer(Layers.inputMixed(features))
            .output(Layers.outputLinearRegression(1));
            
        SimpleNetFloat model = SimpleNet.ofFloatRegression(net);
        
        // The bug: When we train on one item, ALL items would update together
        // But if we're training all items to the SAME target value...
        float commonTarget = 1.0f;
        
        System.out.println("Training all items to the same target (1.0):");
        for (int epoch = 0; epoch < 20; epoch++) {
            // Train each item
            for (int item = 0; item < 3; item++) {
                Map<String, Object> input = new HashMap<>();
                input.put("item", item);
                model.train(input, commonTarget);
            }
            
            if (epoch % 5 == 0) {
                System.out.printf("\nEpoch %d predictions:\n", epoch);
                for (int item = 0; item < 3; item++) {
                    Map<String, Object> input = new HashMap<>();
                    input.put("item", item);
                    System.out.printf("  Item %d: %.3f\n", item, model.predictFloat(input));
                }
            }
        }
        
        // All predictions converge to 1.0 - looks like it's working!
        // But it's actually because ALL embeddings update together
        
        System.out.println("\n=== NOW WITH DIFFERENT TARGETS ===\n");
        
        // Reset with new model
        NeuralNet net2 = NeuralNet.newBuilder()
            .input(features.length)
            .setDefaultOptimizer(new SgdOptimizer(0.1f))
            .layer(Layers.inputMixed(features))
            .output(Layers.outputLinearRegression(1));
            
        SimpleNetFloat model2 = SimpleNet.ofFloatRegression(net2);
        
        // Train with DIFFERENT targets
        float[] targets = {0.1f, 0.5f, 0.9f};
        
        System.out.println("Training items to different targets (0.1, 0.5, 0.9):");
        for (int epoch = 0; epoch < 20; epoch++) {
            for (int item = 0; item < 3; item++) {
                Map<String, Object> input = new HashMap<>();
                input.put("item", item);
                model2.train(input, targets[item]);
            }
            
            if (epoch % 5 == 0) {
                System.out.printf("\nEpoch %d predictions:\n", epoch);
                for (int item = 0; item < 3; item++) {
                    Map<String, Object> input = new HashMap<>();
                    input.put("item", item);
                    System.out.printf("  Item %d: %.3f (target: %.1f)\n", 
                        item, model2.predictFloat(input), targets[item]);
                }
            }
        }
        
        // With the bug, all predictions converge to the AVERAGE (0.5)
        // because all embeddings update together!
    }
    
    @Test
    public void testTypicalTestPattern() {
        System.out.println("\n=== TYPICAL TEST PATTERN ===\n");
        
        // Most tests follow this pattern:
        Feature[] features = {
            Feature.embedding(100, 16, "item"),
            Feature.oneHot(4, "category")
        };
        
        NeuralNet net = NeuralNet.newBuilder()
            .input(features.length)
            .setDefaultOptimizer(new SgdOptimizer(0.01f))
            .layer(Layers.inputMixed(features))
            .layer(Layers.hiddenDenseRelu(32))
            .output(Layers.outputLinearRegression(1));
            
        SimpleNetFloat model = SimpleNet.ofFloatRegression(net);
        
        // Train on random data
        Random rand = new Random(42);
        System.out.println("Training on random data...");
        
        for (int i = 0; i < 1000; i++) {
            Map<String, Object> input = new HashMap<>();
            input.put("item", rand.nextInt(100));
            input.put("category", rand.nextInt(4));
            
            // Random target
            float target = rand.nextFloat();
            model.train(input, target);
        }
        
        // Test: Can it predict something reasonable?
        float totalError = 0;
        for (int i = 0; i < 100; i++) {
            Map<String, Object> input = new HashMap<>();
            input.put("item", rand.nextInt(100));
            input.put("category", rand.nextInt(4));
            
            float pred = model.predictFloat(input);
            float target = rand.nextFloat();
            totalError += Math.abs(pred - target);
        }
        
        float avgError = totalError / 100;
        System.out.printf("Average error: %.3f\n", avgError);
        
        // This test would PASS even with the bug because:
        // 1. With random targets, everything converges to ~0.5
        // 2. Random test targets are also ~0.5 on average
        // 3. So the error looks reasonable!
        
        assertTrue(avgError < 0.5f, "Error should be reasonable");
        System.out.println("Test passed! But the bug is still there...");
    }
    
    @Test
    public void testSparseTrainingRevealsCollapse() {
        System.out.println("\n=== SPARSE TRAINING REVEALS COLLAPSE ===\n");
        
        Feature[] features = {
            Feature.embedding(100, 16, "item")
        };
        
        NeuralNet net = NeuralNet.newBuilder()
            .input(features.length)
            .setDefaultOptimizer(new SgdOptimizer(0.1f))
            .layer(Layers.inputMixed(features))
            .output(Layers.outputLinearRegression(1));
            
        SimpleNetFloat model = SimpleNet.ofFloatRegression(net);
        
        // Train ONLY items 0 and 99 to different values
        System.out.println("Training only items 0 (->0.1) and 99 (->0.9):");
        
        for (int epoch = 0; epoch < 50; epoch++) {
            Map<String, Object> input0 = new HashMap<>();
            input0.put("item", 0);
            model.train(input0, 0.1f);
            
            Map<String, Object> input99 = new HashMap<>();
            input99.put("item", 99);
            model.train(input99, 0.9f);
        }
        
        // Check what happened to untrained items
        System.out.println("\nPredictions after training:");
        for (int item : new int[]{0, 1, 50, 98, 99}) {
            Map<String, Object> input = new HashMap<>();
            input.put("item", item);
            float pred = model.predictFloat(input);
            System.out.printf("  Item %d: %.3f", item, pred);
            if (item == 0 || item == 99) {
                System.out.println(" (trained)");
            } else {
                System.out.println(" (NEVER trained!)");
            }
        }
        
        // With the bug: ALL items converge to ~0.5 (average of 0.1 and 0.9)
        // Without bug: Only items 0 and 99 would change
    }
}