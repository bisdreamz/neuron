package dev.neuronic.net;

import dev.neuronic.net.layers.Feature;
import dev.neuronic.net.layers.HashUtils;
import dev.neuronic.net.optimizers.AdamWOptimizer;
import dev.neuronic.net.simple.SimpleNet;
import dev.neuronic.net.simple.SimpleNetFloat;
import org.junit.jupiter.api.Test;

import java.util.HashMap;
import java.util.Map;

/**
 * Diagnostic test to help users debug why their models predict the same value.
 */
public class DiagnosticTest {
    
    @Test
    public void diagnoseConvergenceIssue() {
        System.out.println("=== DIAGNOSTIC TEST FOR CONVERGENCE TO MEAN ===\n");
        
        // Test 1: One-hot only (should work)
        System.out.println("TEST 1: One-hot only features");
        testConfiguration(
            new Feature[] {
                Feature.oneHot(4, "device"),
                Feature.oneHot(3, "connection")
            },
            "One-hot only"
        );
        
        // Test 2: One-hot + passthrough (should work)
        System.out.println("\nTEST 2: One-hot + passthrough features");
        testConfiguration(
            new Feature[] {
                Feature.oneHot(4, "device"),
                Feature.passthrough("value")
            },
            "One-hot + passthrough"
        );
        
        // Test 3: One-hot + regular embedding (should work)
        System.out.println("\nTEST 3: One-hot + regular embedding");
        testConfiguration(
            new Feature[] {
                Feature.oneHot(4, "device"),
                Feature.embedding(100, 8, "item")
            },
            "One-hot + regular embedding"
        );
        
        // Test 4: One-hot + hashed embedding (likely to fail)
        System.out.println("\nTEST 4: One-hot + hashed embedding");
        testConfiguration(
            new Feature[] {
                Feature.oneHot(4, "device"),
                Feature.hashedEmbedding(1000, 8, "domain")
            },
            "One-hot + hashed embedding"
        );
        
        // Test 5: Hashed embedding only (likely to fail)
        System.out.println("\nTEST 5: Hashed embedding only");
        testConfiguration(
            new Feature[] {
                Feature.hashedEmbedding(1000, 8, "domain"),
                Feature.hashedEmbedding(500, 4, "app")
            },
            "Hashed embedding only"
        );
        
        System.out.println("\n=== DIAGNOSTIC SUMMARY ===");
        System.out.println("If tests 1-3 show learning but 4-5 don't, the issue is with hashed embeddings.");
        System.out.println("If all tests fail to learn, check your data preprocessing.");
        System.out.println("If all tests learn, your actual configuration differs from these tests.");
    }
    
    private void testConfiguration(Feature[] features, String description) {
        AdamWOptimizer optimizer = new AdamWOptimizer(0.01f, 0.0001f);
        
        NeuralNet net = NeuralNet.newBuilder()
            .setDefaultOptimizer(optimizer)
            .layer(Layers.inputMixed(features))
            .layer(Layers.hiddenDenseLeakyRelu(16, 0.01f))
            .output(Layers.outputLinearRegression(1));
            
        SimpleNetFloat model = SimpleNet.ofFloatRegression(net);
        
        // Create simple training data
        Map<String, Object>[] inputs = new Map[4];
        float[] targets = {1.0f, 2.0f, 3.0f, 4.0f};
        
        for (int i = 0; i < 4; i++) {
            inputs[i] = new HashMap<>();
            
            // Fill in values based on feature types
            for (Feature f : features) {
                switch (f.getType()) {
                    case ONEHOT -> inputs[i].put(f.getName(), i % 3); // Use modulo 3 to ensure valid range
                    case PASSTHROUGH -> inputs[i].put(f.getName(), i * 0.5f);
                    case EMBEDDING -> inputs[i].put(f.getName(), i * 10);
                    case HASHED_EMBEDDING -> inputs[i].put(f.getName(), HashUtils.hashString("item_" + i));
                    default -> {} // Skip others
                }
            }
        }
        
        // Initial predictions
        float[] initialPreds = new float[4];
        for (int i = 0; i < 4; i++) {
            initialPreds[i] = model.predictFloat(inputs[i]);
        }
        
        // Train
        for (int epoch = 0; epoch < 50; epoch++) {
            for (int i = 0; i < 4; i++) {
                model.train(inputs[i], targets[i]);
            }
        }
        
        // Final predictions
        float[] finalPreds = new float[4];
        float minPred = Float.MAX_VALUE, maxPred = Float.MIN_VALUE;
        for (int i = 0; i < 4; i++) {
            finalPreds[i] = model.predictFloat(inputs[i]);
            minPred = Math.min(minPred, finalPreds[i]);
            maxPred = Math.max(maxPred, finalPreds[i]);
        }
        
        float range = maxPred - minPred;
        boolean learned = range > 1.0f; // Should have at least 1.0 range for targets 1-4
        
        System.out.printf("%s: Range=%.3f, Learned=%s, Predictions=[%.2f, %.2f, %.2f, %.2f]\n",
                        description, range, learned ? "YES" : "NO", 
                        finalPreds[0], finalPreds[1], finalPreds[2], finalPreds[3]);
    }
}