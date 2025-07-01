package dev.neuronic.net;

import dev.neuronic.net.layers.Feature;
import dev.neuronic.net.optimizers.AdamWOptimizer;
import dev.neuronic.net.simple.SimpleNet;
import dev.neuronic.net.simple.SimpleNetFloat;
import org.junit.jupiter.api.Test;

import java.util.*;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Test to verify mode collapse happens even WITHOUT dropout.
 */
public class ModeCollapseWithoutDropoutTest {
    
    @Test
    public void testModeCollapseWithoutDropout() {
        System.out.println("=== MODE COLLAPSE TEST WITHOUT DROPOUT ===\n");
        
        // Same features as production
        Feature[] features = {
            Feature.hashedEmbedding(50_000, 32, "app_bundle"),
            Feature.embeddingLRU(4000, 16, "zone_id"), 
            Feature.oneHot(25, "country"),
            Feature.oneHot(4, "os"),
            Feature.passthrough("bid_floor")
        };
        
        AdamWOptimizer optimizer = new AdamWOptimizer(0.01f, 0.001f);
        
        // NO DROPOUT in this network
        NeuralNet net = NeuralNet.newBuilder()
            .input(features.length)
            .setDefaultOptimizer(optimizer)
            .layer(Layers.inputMixed(features))
            .layer(Layers.hiddenDenseRelu(128))
            // NO DROPOUT LAYER
            .layer(Layers.hiddenDenseRelu(64))
            .output(Layers.outputLinearRegression(1));
            
        SimpleNetFloat model = SimpleNet.ofFloatRegression(net);
        
        // Train with production-like ratios
        Random rand = new Random(42);
        System.out.println("Training with 1% positive (0.5-3.0) and 2% negative (-0.01)...\n");
        
        int bidCount = 0, negativeCount = 0;
        
        for (int i = 0; i < 50_000; i++) {
            Map<String, Object> request = generateRequest(rand);
            
            if (rand.nextFloat() < 0.01f) {
                // 1% positive samples
                float bidValue = 0.5f + rand.nextFloat() * 2.5f;
                model.train(request, bidValue);
                bidCount++;
            } else if (rand.nextFloat() < 0.02f) {
                // 2% negative samples  
                model.train(request, -0.01f);
                negativeCount++;
            }
            
            // Check for collapse every 10k
            if (i > 0 && i % 10_000 == 0) {
                System.out.printf("Step %d - Bids: %d, Negatives: %d\n", i, bidCount, negativeCount);
                
                // Check prediction diversity
                Set<String> uniquePreds = new HashSet<>();
                float sum = 0;
                float min = Float.MAX_VALUE, max = Float.MIN_VALUE;
                
                for (int j = 0; j < 100; j++) {
                    float pred = model.predictFloat(generateRequest(rand));
                    uniquePreds.add(String.format("%.3f", pred));
                    sum += pred;
                    min = Math.min(min, pred);
                    max = Math.max(max, pred);
                }
                
                System.out.printf("  Unique predictions: %d/100, Range: [%.3f, %.3f], Avg: %.3f\n", 
                    uniquePreds.size(), min, max, sum / 100);
                
                if (uniquePreds.size() < 10) {
                    System.out.println("  ⚠️  MODE COLLAPSE DETECTED!");
                }
                System.out.println();
            }
        }
        
        // Final assessment
        System.out.println("=== FINAL ASSESSMENT ===");
        Set<String> finalUnique = new HashSet<>();
        for (int i = 0; i < 1000; i++) {
            float pred = model.predictFloat(generateRequest(rand));
            finalUnique.add(String.format("%.3f", pred));
        }
        
        System.out.printf("Final diversity: %d unique predictions out of 1000\n", finalUnique.size());
        
        if (finalUnique.size() < 50) {
            System.out.println("\n❌ CONCLUSION: Mode collapse occurs even WITHOUT dropout!");
            System.out.println("   The issue is NOT caused by dropout.");
        } else {
            System.out.println("\n✓ Network maintains diversity without dropout");
        }
    }
    
    @Test 
    public void testDifferentNegativeValues() {
        System.out.println("\n=== TESTING DIFFERENT NEGATIVE VALUES ===\n");
        
        for (float negValue : new float[]{-0.01f, -0.1f, -0.25f, -0.5f}) {
            System.out.printf("Testing with negative value: %.2f\n", negValue);
            testWithNegativeValue(negValue);
            System.out.println();
        }
    }
    
    private void testWithNegativeValue(float negativeValue) {
        Feature[] features = {
            Feature.embeddingLRU(1000, 16, "app"),
            Feature.oneHot(10, "country"),
            Feature.passthrough("value")
        };
        
        AdamWOptimizer optimizer = new AdamWOptimizer(0.01f, 0.001f);
        
        NeuralNet net = NeuralNet.newBuilder()
            .input(features.length)
            .setDefaultOptimizer(optimizer)
            .layer(Layers.inputMixed(features))
            .layer(Layers.hiddenDenseRelu(64))
            .output(Layers.outputLinearRegression(1));
            
        SimpleNetFloat model = SimpleNet.ofFloatRegression(net);
        
        Random rand = new Random(42);
        
        // Quick training
        for (int i = 0; i < 20_000; i++) {
            Map<String, Object> request = new HashMap<>();
            request.put("app", rand.nextInt(1000));
            request.put("country", rand.nextInt(10));
            request.put("value", rand.nextFloat());
            
            if (rand.nextFloat() < 0.01f) {
                model.train(request, 0.5f + rand.nextFloat() * 2.5f);
            } else if (rand.nextFloat() < 0.02f) {
                model.train(request, negativeValue);
            }
        }
        
        // Check diversity
        Set<String> unique = new HashSet<>();
        for (int i = 0; i < 200; i++) {
            Map<String, Object> request = new HashMap<>();
            request.put("app", rand.nextInt(1000));
            request.put("country", rand.nextInt(10));
            request.put("value", rand.nextFloat());
            
            float pred = model.predictFloat(request);
            unique.add(String.format("%.3f", pred));
        }
        
        System.out.printf("  Diversity: %d unique predictions out of 200\n", unique.size());
    }
    
    private Map<String, Object> generateRequest(Random rand) {
        Map<String, Object> request = new HashMap<>();
        request.put("app_bundle", "com.app." + rand.nextInt(50_000));
        request.put("zone_id", rand.nextInt(4000));
        request.put("country", rand.nextInt(20));
        request.put("os", rand.nextInt(4));
        request.put("bid_floor", 0.01f + rand.nextFloat() * 0.49f);
        return request;
    }
}