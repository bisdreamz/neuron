package dev.neuronic.net;

import dev.neuronic.net.layers.Feature;
import dev.neuronic.net.optimizers.AdamWOptimizer;
import dev.neuronic.net.simple.SimpleNet;
import dev.neuronic.net.simple.SimpleNetFloat;
import org.junit.jupiter.api.Test;

import java.util.*;

/**
 * Test if larger networks avoid mode collapse.
 * TODO: REMOVE THIS FILE AFTER DEBUGGING
 */
public class NetworkSizeCollapseTest {
    
    @Test
    public void testLargerNetwork() {
        System.out.println("=== TESTING LARGER NETWORK ===\n");
        
        Feature[] features = {
            Feature.hashedEmbedding(50_000, 32, "app_bundle"),
            Feature.embeddingLRU(4000, 16, "zone_id"),
            Feature.hashedEmbedding(1000, 8, "country"),
            Feature.oneHot(4, "os"),
            Feature.passthrough("bid_floor")
        };
        
        AdamWOptimizer optimizer = new AdamWOptimizer(0.01f, 0.001f);
        
        // MUCH LARGER NETWORK
        NeuralNet net = NeuralNet.newBuilder()
            .input(features.length)
            .setDefaultOptimizer(optimizer)
            .layer(Layers.inputMixed(features))
            .layer(Layers.hiddenDenseRelu(512))  // 4x larger
            .layer(Layers.hiddenDenseRelu(256))  // 4x larger  
            .layer(Layers.hiddenDenseRelu(128))  // Extra layer
            .output(Layers.outputLinearRegression(1));
            
        SimpleNetFloat model = SimpleNet.ofFloatRegression(net);
        
        System.out.println("Network: Input(~100) -> 512 -> 256 -> 128 -> 1");
        System.out.println("Total parameters: ~200k+ (vs ~25k in original)");
        System.out.println();
        
        testCollapseScenario(model, "LARGE NETWORK");
    }
    
    @Test
    public void testOriginalNetwork() {
        System.out.println("=== TESTING ORIGINAL NETWORK ===\n");
        
        Feature[] features = {
            Feature.hashedEmbedding(50_000, 32, "app_bundle"),
            Feature.embeddingLRU(4000, 16, "zone_id"),
            Feature.hashedEmbedding(1000, 8, "country"),
            Feature.oneHot(4, "os"),
            Feature.passthrough("bid_floor")
        };
        
        AdamWOptimizer optimizer = new AdamWOptimizer(0.01f, 0.001f);
        
        // ORIGINAL NETWORK
        NeuralNet net = NeuralNet.newBuilder()
            .input(features.length)
            .setDefaultOptimizer(optimizer)
            .layer(Layers.inputMixed(features))
            .layer(Layers.hiddenDenseRelu(128))
            .layer(Layers.hiddenDenseRelu(64))
            .output(Layers.outputLinearRegression(1));
            
        SimpleNetFloat model = SimpleNet.ofFloatRegression(net);
        
        System.out.println("Network: Input(~100) -> 128 -> 64 -> 1");
        System.out.println();
        
        testCollapseScenario(model, "ORIGINAL NETWORK");
    }
    
    private void testCollapseScenario(SimpleNetFloat model, String networkType) {
        Set<String> goodApps = new HashSet<>();
        for (int i = 0; i < 50; i++) {
            goodApps.add("com.premium.app" + i);
        }
        
        Set<Integer> goodZones = new HashSet<>();
        for (int i = 0; i < 200; i++) {
            goodZones.add(i);
        }
        
        Random rand = new Random(42);
        
        // Track collapse during training
        for (int step = 0; step < 6000; step++) {
            boolean isGoodSegment = rand.nextBoolean();
            
            Map<String, Object> request = new HashMap<>();
            
            if (isGoodSegment) {
                request.put("app_bundle", goodApps.toArray()[rand.nextInt(goodApps.size())]);
                request.put("zone_id", goodZones.toArray()[rand.nextInt(goodZones.size())]);
                request.put("country", "US");
                request.put("os", "ios");
            } else {
                request.put("app_bundle", "com.junk.app" + rand.nextInt(10_000));
                request.put("zone_id", 1000 + rand.nextInt(3000));
                request.put("country", "country_" + rand.nextInt(25));
                request.put("os", "android");
            }
            request.put("bid_floor", 0.01f + rand.nextFloat() * 0.5f);
            
            // Train with 2% probability
            if (rand.nextFloat() < 0.02f) {
                if (isGoodSegment) {
                    float value = 0.5f + rand.nextFloat() * 2.5f;
                    model.train(request, value);
                } else {
                    model.train(request, -0.25f);
                }
            }
            
            // Check collapse every 2000 steps
            if (step > 0 && step % 2000 == 0) {
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
                
                boolean collapsed = uniqueGood.size() == 1 && uniqueBad.size() == 1;
                System.out.printf("%s Step %d: Good unique=%d, Bad unique=%d %s\n", 
                    networkType, step, uniqueGood.size(), uniqueBad.size(),
                    collapsed ? "⚠️ COLLAPSED!" : "✓");
                
                if (collapsed) {
                    System.out.printf("  Collapse values: Good=%.3f, Bad=%.3f\n", 
                        goodSum/100, badSum/100);
                }
            }
        }
        System.out.println();
    }
}