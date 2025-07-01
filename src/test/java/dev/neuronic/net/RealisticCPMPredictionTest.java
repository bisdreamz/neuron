package dev.neuronic.net;

import dev.neuronic.net.layers.Feature;
import dev.neuronic.net.optimizers.AdamWOptimizer;
import dev.neuronic.net.simple.SimpleNet;
import dev.neuronic.net.simple.SimpleNetFloat;
import org.junit.jupiter.api.Test;

import java.util.*;

/**
 * Realistic CPM prediction test matching production scenario:
 * - 7 features with proper cardinality
 * - 5% of segments see 80% of positive bids
 * - Negative samples for segments that don't monetize
 * - Validate accurate CPM prediction, not just averages
 */
public class RealisticCPMPredictionTest {
    
    @Test
    public void testRealisticCPMPrediction() {
        System.out.println("=== REALISTIC CPM PREDICTION TEST ===\n");
        
        // Your exact feature setup
        Feature[] features = {
            Feature.oneHot(10, "os"),                        // iOS, Android, Windows, etc.
            Feature.embeddingLRU(100, 8, "pubid"),          // 100 publishers
            Feature.hashedEmbedding(10000, 16, "app_bundle"), // 1000s of apps (hashed for scale)
            Feature.embeddingLRU(4000, 12, "zone_id"),      // 4000 ad zones
            Feature.oneHot(7, "device_type"),               // Phone, Tablet, TV, etc.
            Feature.oneHot(5, "connection_type"),           // WiFi, 4G, 3G, etc.
            Feature.passthrough("bid_floor")                // Minimum bid price
        };
        
        // Test with different configurations
        testConfiguration(features, 0.01f, 10.0f, "LR=0.01, clip=10");
        testConfiguration(features, 0.001f, 1.0f, "LR=0.001, clip=1");
    }
    
    private void testConfiguration(Feature[] features, float lr, float clipNorm, String configName) {
        System.out.printf("\n=== Testing %s ===\n", configName);
        
        AdamWOptimizer optimizer = new AdamWOptimizer(lr, 0.001f);
        
        NeuralNet net = NeuralNet.newBuilder()
            .input(features.length)
            .setDefaultOptimizer(optimizer)
            .layer(Layers.inputMixed(features))
            .layer(Layers.hiddenDenseRelu(256))  // Larger capacity
            .layer(Layers.hiddenDenseRelu(128))
            .layer(Layers.hiddenDenseRelu(64))
            .withGlobalGradientClipping(clipNorm)
            .output(Layers.outputLinearRegression(1));
            
        SimpleNetFloat model = SimpleNet.ofFloatRegression(net);
        
        Random rand = new Random(42);
        
        // Create segment quality mapping
        // Premium segments (5% of combinations see 80% of bids)
        Set<String> premiumSegments = new HashSet<>();
        Map<String, Float> segmentBaseCPM = new HashMap<>();
        
        // Define premium publishers and apps
        Set<Integer> premiumPubs = new HashSet<>(Arrays.asList(1, 2, 3, 4, 5)); // Top 5 publishers
        Set<String> premiumApps = new HashSet<>();
        for (int i = 0; i < 50; i++) {
            premiumApps.add("com.premium.app" + i);
        }
        Set<Integer> premiumZones = new HashSet<>();
        for (int i = 0; i < 200; i++) {
            premiumZones.add(i);
        }
        
        // Generate test segments with known CPM values
        List<Map<String, Object>> testSegments = new ArrayList<>();
        List<Float> expectedCPMs = new ArrayList<>();
        List<String> segmentDescriptions = new ArrayList<>();
        
        // Premium segments
        for (int i = 0; i < 10; i++) {
            Map<String, Object> segment = new HashMap<>();
            segment.put("os", i < 5 ? "ios" : "android");
            segment.put("pubid", 1 + i % 5);  // Premium publishers
            segment.put("app_bundle", "com.premium.app" + i);
            segment.put("zone_id", i);
            segment.put("device_type", i < 7 ? "phone" : "tablet");
            segment.put("connection_type", "wifi");
            segment.put("bid_floor", 1.0f + i * 0.1f);
            
            testSegments.add(segment);
            expectedCPMs.add(3.0f + rand.nextFloat() * 2.0f); // $3-5 CPM
            segmentDescriptions.add(String.format("Premium %d", i));
        }
        
        // Good segments
        for (int i = 0; i < 10; i++) {
            Map<String, Object> segment = new HashMap<>();
            segment.put("os", "android");
            segment.put("pubid", 10 + i);
            segment.put("app_bundle", "com.good.app" + i);
            segment.put("zone_id", 200 + i);
            segment.put("device_type", "phone");
            segment.put("connection_type", i < 5 ? "wifi" : "4g");
            segment.put("bid_floor", 0.5f + i * 0.05f);
            
            testSegments.add(segment);
            expectedCPMs.add(1.0f + rand.nextFloat() * 1.0f); // $1-2 CPM
            segmentDescriptions.add(String.format("Good %d", i));
        }
        
        // Poor segments
        for (int i = 0; i < 10; i++) {
            Map<String, Object> segment = new HashMap<>();
            segment.put("os", "android");
            segment.put("pubid", 50 + i);
            segment.put("app_bundle", "com.unknown.app" + i);
            segment.put("zone_id", 3000 + i);
            segment.put("device_type", "phone");
            segment.put("connection_type", "3g");
            segment.put("bid_floor", 0.01f + i * 0.01f);
            
            testSegments.add(segment);
            expectedCPMs.add(0.1f + rand.nextFloat() * 0.4f); // $0.10-0.50 CPM
            segmentDescriptions.add(String.format("Poor %d", i));
        }
        
        System.out.println("Training with realistic bid distribution...");
        System.out.println("- 5% premium segments get 80% of positive bids");
        System.out.println("- Many segments get negative/zero bids (no monetization)");
        
        // Training simulation
        int premiumBids = 0, regularBids = 0, negativeSamples = 0;
        
        for (int step = 0; step < 10000; step++) {
            Map<String, Object> input = new HashMap<>();
            float target;
            
            float segmentDraw = rand.nextFloat();
            
            if (segmentDraw < 0.8f) { // 80% chance of premium segment bid
                // Premium segment
                input.put("os", rand.nextBoolean() ? "ios" : "android");
                input.put("pubid", 1 + rand.nextInt(5));
                input.put("app_bundle", "com.premium.app" + rand.nextInt(50));
                input.put("zone_id", rand.nextInt(200));
                input.put("device_type", "phone");
                input.put("connection_type", "wifi");
                input.put("bid_floor", 1.0f + rand.nextFloat());
                
                // High CPM with variance
                target = 3.0f + rand.nextFloat() * 2.0f; // $3-5
                premiumBids++;
                
            } else if (segmentDraw < 0.95f) { // 15% regular segments
                // Regular segment
                input.put("os", "android");
                input.put("pubid", 10 + rand.nextInt(40));
                input.put("app_bundle", "com.app" + rand.nextInt(500));
                input.put("zone_id", 200 + rand.nextInt(1800));
                input.put("device_type", rand.nextBoolean() ? "phone" : "tablet");
                input.put("connection_type", rand.nextBoolean() ? "wifi" : "4g");
                input.put("bid_floor", 0.1f + rand.nextFloat() * 0.5f);
                
                // Medium CPM
                target = 0.5f + rand.nextFloat() * 1.5f; // $0.50-2
                regularBids++;
                
            } else { // 5% negative samples (no bids)
                // Random segment that doesn't monetize
                input.put("os", "android");
                input.put("pubid", 50 + rand.nextInt(50));
                input.put("app_bundle", "com.junk.app" + rand.nextInt(500));
                input.put("zone_id", 2000 + rand.nextInt(2000));
                input.put("device_type", "phone");
                input.put("connection_type", "3g");
                input.put("bid_floor", 0.01f + rand.nextFloat() * 0.1f);
                
                // Zero or very low CPM (no monetization)
                target = rand.nextFloat() < 0.8f ? 0.0f : 0.01f + rand.nextFloat() * 0.09f;
                negativeSamples++;
            }

            model.train(input, target);
            
            // Evaluate periodically
            if (step > 0 && step % 2000 == 0) {
                System.out.printf("\nStep %d - Trained: %d premium, %d regular, %d negative\n",
                    step, premiumBids, regularBids, negativeSamples);
                evaluatePredictions(model, testSegments, expectedCPMs, segmentDescriptions);
            }
        }
        
        // Final evaluation
        System.out.println("\n=== FINAL EVALUATION ===");
        System.out.printf("Total training: %d premium bids (%.1f%%), %d regular (%.1f%%), %d negative (%.1f%%)\n",
            premiumBids, premiumBids/100.0, regularBids, regularBids/100.0, negativeSamples, negativeSamples/100.0);
        
        float avgError = evaluatePredictions(model, testSegments, expectedCPMs, segmentDescriptions);
        
        // Test generalization on unseen segments
        System.out.println("\nGeneralization test (completely new segments):");
        float generalizationError = 0;
        int correctRankings = 0;
        
        for (int i = 0; i < 10; i++) {
            // New premium segment
            Map<String, Object> premium = new HashMap<>();
            premium.put("os", "ios");
            premium.put("pubid", 1); // Known premium publisher
            premium.put("app_bundle", "com.new.premium" + i); // New app
            premium.put("zone_id", 50 + i); // New zone
            premium.put("device_type", "phone");
            premium.put("connection_type", "wifi");
            premium.put("bid_floor", 2.0f);
            
            // New poor segment
            Map<String, Object> poor = new HashMap<>();
            poor.put("os", "android");
            poor.put("pubid", 90); // Unknown publisher
            poor.put("app_bundle", "com.new.unknown" + i);
            poor.put("zone_id", 3500 + i);
            poor.put("device_type", "phone");
            poor.put("connection_type", "3g");
            poor.put("bid_floor", 0.05f);
            
            float premiumPred = model.predictFloat(premium);
            float poorPred = model.predictFloat(poor);
            
            System.out.printf("  New premium %d: $%.2f CPM\n", i, premiumPred);
            System.out.printf("  New poor %d: $%.2f CPM\n", i, poorPred);
            
            if (premiumPred > poorPred) correctRankings++;
            
            // Premium should be $3-5, poor should be $0.1-0.5
            generalizationError += Math.abs(premiumPred - 4.0f);
            generalizationError += Math.abs(poorPred - 0.3f);
        }
        
        generalizationError /= 20;
        System.out.printf("\nGeneralization error: $%.3f\n", generalizationError);
        System.out.printf("Correct rankings: %d/10\n", correctRankings);
        
        // Success criteria
        boolean success = avgError < 0.5f && correctRankings >= 8 && generalizationError < 1.0f;
        System.out.printf("\nResult: %s\n", success ? 
            "✓ SUCCESS - Network learns accurate CPMs for different segments!" :
            "✗ FAILURE - Network cannot differentiate segments properly!");
    }
    
    private float evaluatePredictions(SimpleNetFloat model, List<Map<String, Object>> testSegments,
                                     List<Float> expectedCPMs, List<String> descriptions) {
        System.out.println("Segment predictions:");
        
        float totalError = 0;
        Set<String> uniquePreds = new HashSet<>();
        
        for (int i = 0; i < testSegments.size(); i++) {
            float pred = model.predictFloat(testSegments.get(i));
            float expected = expectedCPMs.get(i);
            float error = Math.abs(pred - expected);
            
            totalError += error;
            uniquePreds.add(String.format("%.2f", pred));
            
            System.out.printf("  %s: expected=$%.2f, predicted=$%.2f, error=$%.2f %s\n",
                descriptions.get(i), expected, pred, error,
                error > 1.0f ? "⚠️" : "");
        }
        
        float avgError = totalError / testSegments.size();
        System.out.printf("Average error: $%.3f, Unique predictions: %d/%d\n", 
            avgError, uniquePreds.size(), testSegments.size());
        
        if (uniquePreds.size() < 5) {
            System.out.println("⚠️ WARNING: Low prediction diversity - possible collapse!");
        }
        
        return avgError;
    }
}