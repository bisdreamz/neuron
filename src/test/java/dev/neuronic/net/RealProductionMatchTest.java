package dev.neuronic.net;

import dev.neuronic.net.layers.Feature;
import dev.neuronic.net.optimizers.AdamWOptimizer;
import dev.neuronic.net.simple.SimpleNet;
import dev.neuronic.net.simple.SimpleNetFloat;
import org.junit.jupiter.api.Test;

import java.util.*;

/**
 * Test that EXACTLY matches your real production setup.
 * TODO: REMOVE THIS FILE AFTER DEBUGGING
 */
public class RealProductionMatchTest {
    
    @Test
    public void testExactProductionSetup() {
        System.out.println("=== EXACT PRODUCTION SETUP ===\n");
        
        // YOUR ACTUAL FEATURES - LRU embeddings, NOT hashed
        Feature[] features = {
            Feature.embeddingLRU(50_000, 32, "app_bundle"),  // LRU not hashed!
            Feature.embeddingLRU(4000, 16, "zone_id"),       // LRU 
            Feature.oneHot(25, "country"),                    // oneHot not hashed
            Feature.oneHot(4, "os"),
            Feature.passthrough("bid_floor")
        };
        
        AdamWOptimizer optimizer = new AdamWOptimizer(0.01f, 0.001f);
        
        NeuralNet net = NeuralNet.newBuilder()
            .input(features.length)
            .setDefaultOptimizer(optimizer)
            .layer(Layers.inputMixed(features))
            .layer(Layers.hiddenDenseRelu(128))
            .layer(Layers.hiddenDenseRelu(64))
            .output(Layers.outputLinearRegression(1));
            
        SimpleNetFloat model = SimpleNet.ofFloatRegression(net);
        
        System.out.println("Features: embeddingLRU + embeddingLRU + oneHot + oneHot + passthrough");
        System.out.println("Training: 2% sparse rate with good/bad segments");
        System.out.println();
        
        Set<String> goodApps = new HashSet<>();
        for (int i = 0; i < 50; i++) {
            goodApps.add("com.premium.app" + i);
        }
        
        Set<Integer> goodZones = new HashSet<>();
        for (int i = 0; i < 200; i++) {
            goodZones.add(i);
        }
        
        Random rand = new Random(42);
        
        int positiveCount = 0;
        int negativeCount = 0;
        
        // Track predictions over time
        for (int step = 0; step < 8000; step++) {
            // Generate request
            boolean isGoodSegment = rand.nextBoolean();
            
            Map<String, Object> request = new HashMap<>();
            
            if (isGoodSegment) {
                // Good segment
                request.put("app_bundle", goodApps.toArray()[rand.nextInt(goodApps.size())]);
                request.put("zone_id", goodZones.toArray()[rand.nextInt(goodZones.size())]);
                request.put("country", "US");
                request.put("os", "ios");
            } else {
                // Bad segment  
                request.put("app_bundle", "com.junk.app" + rand.nextInt(10_000));
                request.put("zone_id", 1000 + rand.nextInt(3000));
                request.put("country", rand.nextInt(25));  // Use int for oneHot
                request.put("os", rand.nextInt(4));        // Use int for oneHot
            }
            request.put("bid_floor", 0.01f + rand.nextFloat() * 0.5f);
            
            // Train with 2% probability (YOUR ACTUAL RATE)
            if (rand.nextFloat() < 0.02f) {
                if (isGoodSegment) {
                    // Positive training
                    float value = 0.5f + rand.nextFloat() * 2.5f;
                    model.train(request, value);
                    positiveCount++;
                } else {
                    // Negative training
                    model.train(request, -0.25f);
                    negativeCount++;
                }
            }
            
            // Check predictions every 2000 steps
            if (step > 0 && step % 2000 == 0) {
                System.out.printf("Step %d - Trained %d positive, %d negative\\n", 
                    step, positiveCount, negativeCount);
                
                // Test predictions on known good vs bad
                float goodSum = 0, badSum = 0;
                Set<String> uniqueGood = new HashSet<>();
                Set<String> uniqueBad = new HashSet<>();
                
                for (int i = 0; i < 100; i++) {
                    // Good segment prediction
                    Map<String, Object> goodReq = new HashMap<>();
                    goodReq.put("app_bundle", "com.premium.app0");
                    goodReq.put("zone_id", 0);
                    goodReq.put("country", "US");
                    goodReq.put("os", "ios");
                    goodReq.put("bid_floor", 0.1f);
                    float goodPred = model.predictFloat(goodReq);
                    goodSum += goodPred;
                    uniqueGood.add(String.format("%.3f", goodPred));
                    
                    // Bad segment prediction
                    Map<String, Object> badReq = new HashMap<>();
                    badReq.put("app_bundle", "com.junk.app9999");
                    badReq.put("zone_id", 3999);
                    badReq.put("country", 20);  // Different country
                    badReq.put("os", 2);        // Different OS
                    badReq.put("bid_floor", 0.1f);
                    float badPred = model.predictFloat(badReq);
                    badSum += badPred;
                    uniqueBad.add(String.format("%.3f", badPred));
                }
                
                System.out.printf("  Good segments: avg=%.3f, unique=%d\\n", goodSum/100, uniqueGood.size());
                System.out.printf("  Bad segments: avg=%.3f, unique=%d\\n", badSum/100, uniqueBad.size());
                System.out.printf("  Discrimination: %.3f\\n", (goodSum - badSum)/100);
                
                if (uniqueGood.size() == 1 && uniqueBad.size() == 1) {
                    System.out.println("  ⚠️  COLLAPSED TO CONSTANT!");
                } else if (uniqueGood.size() < 5 || uniqueBad.size() < 5) {
                    System.out.println("  ⚠️  LOW DIVERSITY!");
                } else {
                    System.out.println("  ✓  Healthy diversity");
                }
                System.out.println();
            }
        }
    }
}