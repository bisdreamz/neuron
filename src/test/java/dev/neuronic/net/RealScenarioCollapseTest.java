package dev.neuronic.net;

import dev.neuronic.net.layers.Feature;
import dev.neuronic.net.optimizers.AdamWOptimizer;
import dev.neuronic.net.simple.SimpleNet;
import dev.neuronic.net.simple.SimpleNetFloat;
import org.junit.jupiter.api.Test;

import java.util.*;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Test that ACTUALLY mimics the real production scenario.
 */
public class RealScenarioCollapseTest {
    
    @Test
    public void testRealProductionScenario() {
        System.out.println("=== REAL PRODUCTION SCENARIO ===\n");
        
        // Your actual features
        Feature[] features = {
            Feature.hashedEmbedding(50_000, 32, "app_bundle"),
            Feature.embeddingLRU(4000, 16, "zone_id"),
            Feature.hashedEmbedding(1000, 8, "country"),  // 1000 buckets for 8-dim
            Feature.oneHot(4, "os"),
            Feature.passthrough("bid_floor")
        };
        
        AdamWOptimizer optimizer = new AdamWOptimizer(0.001f, 0.001f); // Lower LR
        
        NeuralNet net = NeuralNet.newBuilder()
            .input(features.length)
            .setDefaultOptimizer(optimizer)
            .layer(Layers.inputMixed(features))
            .layer(Layers.hiddenDenseRelu(128))
            .layer(Layers.hiddenDenseRelu(64))
            .output(Layers.outputLinearRegression(1));
            
        SimpleNetFloat model = SimpleNet.ofFloatRegression(net);
        
        // Define good vs bad segments
        Set<String> goodApps = new HashSet<>();
        for (int i = 0; i < 50; i++) {
            goodApps.add("com.premium.app" + i);
        }
        
        Set<Integer> goodZones = new HashSet<>();
        for (int i = 0; i < 200; i++) {
            goodZones.add(i);
        }
        
        Random rand = new Random(42);
        
        System.out.println("Training with EQUAL positive/negative samples:");
        System.out.println("- Good segments (premium apps/zones) -> positive values (0.5-3.0)");
        System.out.println("- Bad segments -> negative values (-0.25)");
        System.out.println("- Equal training rate for both\n");
        
        int positiveCount = 0;
        int negativeCount = 0;
        
        // Track predictions over time
        for (int step = 0; step < 10_000; step++) {
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
                request.put("country", "country_" + rand.nextInt(25));
                request.put("os", "android");  // Use string for oneHot features
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
                System.out.printf("\nStep %d - Trained %d positive, %d negative (ratio %.1f:1)\n", 
                    step, positiveCount, negativeCount, 
                    negativeCount > 0 ? (float)positiveCount/negativeCount : 0);
                
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
                    badReq.put("country", "country_" + rand.nextInt(25));
                    badReq.put("os", "android");
                    badReq.put("bid_floor", 0.1f);
                    float badPred = model.predictFloat(badReq);
                    badSum += badPred;
                    uniqueBad.add(String.format("%.3f", badPred));
                }
                
                System.out.printf("  Good segments: avg=%.3f, unique=%d\n", goodSum/100, uniqueGood.size());
                System.out.printf("  Bad segments: avg=%.3f, unique=%d\n", badSum/100, uniqueBad.size());
                System.out.printf("  Discrimination: %.3f\n", (goodSum - badSum)/100);
                
                if (uniqueGood.size() == 1 && uniqueBad.size() == 1) {
                    System.out.println("  ⚠️  COLLAPSED TO CONSTANT!");
                }
            }
        }
        
        // Final test
        System.out.println("\n=== FINAL TEST ===");
        testDiscrimination(model, goodApps, goodZones);
    }
    
    private void testDiscrimination(SimpleNetFloat model, Set<String> goodApps, Set<Integer> goodZones) {
        Random rand = new Random(123);
        
        // Test 1000 samples
        List<Float> goodPreds = new ArrayList<>();
        List<Float> badPreds = new ArrayList<>();
        
        for (int i = 0; i < 500; i++) {
            // Good segment
            Map<String, Object> goodReq = new HashMap<>();
            goodReq.put("app_bundle", goodApps.toArray()[i % goodApps.size()]);
            goodReq.put("zone_id", goodZones.toArray()[i % goodZones.size()]);
            goodReq.put("country", "US");
            goodReq.put("os", "ios");
            goodReq.put("bid_floor", 0.1f);
            goodPreds.add(model.predictFloat(goodReq));
            
            // Bad segment
            Map<String, Object> badReq = new HashMap<>();
            badReq.put("app_bundle", "com.junk.app" + (5000 + i));
            badReq.put("zone_id", 2000 + i);
            badReq.put("country", "country_" + rand.nextInt(25));
            badReq.put("os", "android");
            badReq.put("bid_floor", 0.1f);
            badPreds.add(model.predictFloat(badReq));
        }
        
        // Calculate stats
        float goodAvg = goodPreds.stream().reduce(0f, Float::sum) / goodPreds.size();
        float badAvg = badPreds.stream().reduce(0f, Float::sum) / badPreds.size();
        
        Set<String> uniqueGood = new HashSet<>();
        Set<String> uniqueBad = new HashSet<>();
        for (float p : goodPreds) uniqueGood.add(String.format("%.3f", p));
        for (float p : badPreds) uniqueBad.add(String.format("%.3f", p));
        
        System.out.printf("Good segments: avg=%.3f, unique values=%d\n", goodAvg, uniqueGood.size());
        System.out.printf("Bad segments: avg=%.3f, unique values=%d\n", badAvg, uniqueBad.size());
        System.out.printf("Discrimination: %.3f (should be > 0.5)\n", goodAvg - badAvg);
        
        if (uniqueGood.size() < 10 || uniqueBad.size() < 10) {
            System.out.println("\n❌ FAILURE: Model has collapsed!");
            System.out.println("   Not enough unique predictions.");
        } else if (Math.abs(goodAvg - badAvg) < 0.1f) {
            System.out.println("\n❌ FAILURE: No discrimination!");
            System.out.println("   Good and bad segments have similar predictions.");
        } else {
            System.out.println("\n✓ Success: Model can discriminate between segments.");
        }
    }
}