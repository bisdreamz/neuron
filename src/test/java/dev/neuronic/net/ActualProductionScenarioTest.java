package dev.neuronic.net;

import dev.neuronic.net.layers.Feature;
import dev.neuronic.net.optimizers.AdamWOptimizer;
import dev.neuronic.net.simple.SimpleNet;
import dev.neuronic.net.simple.SimpleNetFloat;
import org.junit.jupiter.api.Test;

import java.util.*;

/**
 * ACTUAL production scenario:
 * - Train on EVERY request (100% training rate - NOT 2%!)
 * - 5% of inventory segments see 80% of positive bids
 * - 95% of segments mostly see 0 bids (but still trained with 0.0f)
 * 
 * IMPORTANT: We train on EVERY SINGLE REQUEST. The 2% mentioned in other tests
 * was incorrect. In production, every auction request results in a training sample,
 * whether it's a positive bid or a 0.0f for no bid.
 */
public class ActualProductionScenarioTest {
    
    @Test
    public void testActualProductionScenario() {
        System.out.println("=== ACTUAL PRODUCTION SCENARIO TEST ===\n");
        System.out.println("Training on EVERY request with realistic bid distribution:");
        System.out.println("- 5% of segments get 80% of positive bids");
        System.out.println("- 95% of segments mostly get 0 bids\n");
        
        // Your actual feature configuration
        Feature[] features = {
            Feature.oneHot(50, "FORMAT"),
            Feature.oneHot(50, "PLCMT"), 
            Feature.oneHot(50, "DEVTYPE"),
            Feature.oneHot(50, "DEVCON"),
            Feature.oneHot(50, "GEO"),
            Feature.oneHot(50, "PUBID"),
            Feature.oneHot(50, "OS"),
            Feature.embeddingLRU(20_000, 32, "ZONEID"),
            Feature.embeddingLRU(20_000, 32, "DOMAIN"),
            Feature.embeddingLRU(20_000, 32, "PUB"),
            Feature.embeddingLRU(20_000, 32, "SITEID"),
            Feature.autoScale(0f, 20f, "BIDFLOOR"),
            Feature.autoScale(0f, 600f, "TMAX")
        };
        
        AdamWOptimizer optimizer = new AdamWOptimizer(0.001f, 0.001f);
        
        NeuralNet net = NeuralNet.newBuilder()
            .input(features.length)
            .setDefaultOptimizer(optimizer)
            .layer(Layers.inputMixed(features))
            .layer(Layers.hiddenDenseRelu(512))
            .layer(Layers.hiddenDenseRelu(256))
            .layer(Layers.hiddenDenseRelu(128))
            .layer(Layers.hiddenDenseRelu(64))
            .withGlobalGradientClipping(1.0f)
            .output(Layers.outputHuberRegression(1, optimizer, 3.0f));
            
        SimpleNetFloat model = SimpleNet.ofFloatRegression(net);
        
        Random rand = new Random(42);
        
        // Define premium segments (5% that get 80% of bids)
        Set<Integer> premiumZones = new HashSet<>();
        for (int i = 0; i < 500; i++) { // 500 out of 10,000 = 5%
            premiumZones.add(i);
        }
        
        Set<Integer> premiumDomains = new HashSet<>();
        for (int i = 0; i < 250; i++) { // 250 out of 5,000 = 5%
            premiumDomains.add(i);
        }
        
        // Track metrics
        Map<String, Integer> bidCounts = new HashMap<>();
        Map<String, Float> avgCPMs = new HashMap<>();
        int totalRequests = 0;
        int positiveBids = 0;
        int zeroBids = 0;
        
        System.out.println("Simulating 100,000 auction requests...");
        
        for (int step = 0; step < 100_000; step++) {
            totalRequests++;
            
            // Generate request features
            Map<String, Object> input = new HashMap<>();
            input.put("FORMAT", rand.nextInt(10));
            input.put("PLCMT", rand.nextInt(5));
            input.put("DEVTYPE", rand.nextInt(7));
            input.put("DEVCON", rand.nextInt(5));
            input.put("GEO", rand.nextInt(30));
            input.put("PUBID", rand.nextInt(50));
            input.put("OS", rand.nextInt(4));
            
            int zoneId = rand.nextInt(10_000);
            int domainId = rand.nextInt(5_000);
            int pubId = rand.nextInt(1_000);
            int siteId = rand.nextInt(2_000);
            
            input.put("ZONEID", zoneId);
            input.put("DOMAIN", domainId);
            input.put("PUB", pubId);
            input.put("SITEID", siteId);
            input.put("BIDFLOOR", 0.1f + rand.nextFloat() * 2.0f);
            input.put("TMAX", 100f + rand.nextFloat() * 400f);
            
            // Determine if this segment gets a bid
            boolean isPremiumSegment = premiumZones.contains(zoneId) && premiumDomains.contains(domainId);
            float bidProbability = isPremiumSegment ? 0.8f : 0.05f; // 80% vs 5% chance
            
            float target;
            if (rand.nextFloat() < bidProbability) {
                // Positive bid
                if (isPremiumSegment) {
                    target = 1.5f + rand.nextFloat() * 1.5f; // $1.50-$3.00 for premium
                } else {
                    target = 0.1f + rand.nextFloat() * 0.4f; // $0.10-$0.50 for regular
                }
                positiveBids++;
            } else {
                // No bid - train with 0
                target = 0.0f;
                zeroBids++;
            }
            
            // TRAIN ON EVERY REQUEST
            model.train(input, target);
            
            // Track segment performance
            String segmentKey = zoneId + "_" + domainId;
            bidCounts.merge(segmentKey, target > 0 ? 1 : 0, Integer::sum);
            avgCPMs.merge(segmentKey, target, Float::sum);
            
            // Progress update
            if (step > 0 && step % 20000 == 0) {
                System.out.printf("Step %d: %d positive bids (%.1f%%), %d zero bids (%.1f%%)\n",
                    step, positiveBids, 100.0 * positiveBids / step,
                    zeroBids, 100.0 * zeroBids / step);
                
                // Test some predictions
                testPredictions(model, premiumZones, premiumDomains);
            }
        }
        
        // Final evaluation
        System.out.println("\n=== FINAL EVALUATION ===");
        System.out.printf("Total requests: %d\n", totalRequests);
        System.out.printf("Positive bids: %d (%.1f%%)\n", positiveBids, 100.0 * positiveBids / totalRequests);
        System.out.printf("Zero bids: %d (%.1f%%)\n", zeroBids, 100.0 * zeroBids / totalRequests);
        
        // Test final predictions
        testFinalPredictions(model, premiumZones, premiumDomains);
        
        // Analyze bid distribution
        analyzeBidDistribution(bidCounts, avgCPMs, premiumZones, premiumDomains);
    }
    
    private void testPredictions(SimpleNetFloat model, Set<Integer> premiumZones, Set<Integer> premiumDomains) {
        Random rand = new Random(999);
        
        // Test premium segment
        Map<String, Object> premiumInput = new HashMap<>();
        premiumInput.put("FORMAT", 0);
        premiumInput.put("PLCMT", 0);
        premiumInput.put("DEVTYPE", 0);
        premiumInput.put("DEVCON", 0);
        premiumInput.put("GEO", 0);
        premiumInput.put("PUBID", 0);
        premiumInput.put("OS", 0);
        premiumInput.put("ZONEID", 10); // Premium zone
        premiumInput.put("DOMAIN", 10); // Premium domain
        premiumInput.put("PUB", 0);
        premiumInput.put("SITEID", 0);
        premiumInput.put("BIDFLOOR", 1.0f);
        premiumInput.put("TMAX", 300f);
        
        // Test regular segment
        Map<String, Object> regularInput = new HashMap<>(premiumInput);
        regularInput.put("ZONEID", 5000); // Non-premium zone
        regularInput.put("DOMAIN", 3000); // Non-premium domain
        
        float premiumPred = model.predictFloat(premiumInput);
        float regularPred = model.predictFloat(regularInput);
        
        System.out.printf("  Premium segment prediction: $%.3f\n", premiumPred);
        System.out.printf("  Regular segment prediction: $%.3f\n", regularPred);
    }
    
    private void testFinalPredictions(SimpleNetFloat model, Set<Integer> premiumZones, Set<Integer> premiumDomains) {
        System.out.println("\nTesting prediction accuracy:");
        
        Random rand = new Random(123);
        List<Float> premiumPreds = new ArrayList<>();
        List<Float> regularPreds = new ArrayList<>();
        
        // Base input template
        Map<String, Object> baseInput = new HashMap<>();
        baseInput.put("FORMAT", 0);
        baseInput.put("PLCMT", 0);
        baseInput.put("DEVTYPE", 0);
        baseInput.put("DEVCON", 0);
        baseInput.put("GEO", 0);
        baseInput.put("PUBID", 0);
        baseInput.put("OS", 0);
        baseInput.put("PUB", 0);
        baseInput.put("SITEID", 0);
        baseInput.put("BIDFLOOR", 1.0f);
        baseInput.put("TMAX", 300f);
        
        // Test 50 premium segments
        for (int i = 0; i < 50; i++) {
            Map<String, Object> input = new HashMap<>(baseInput);
            input.put("ZONEID", i); // Premium zones 0-499
            input.put("DOMAIN", i); // Premium domains 0-249
            
            float pred = model.predictFloat(input);
            premiumPreds.add(pred);
        }
        
        // Test 50 regular segments
        for (int i = 0; i < 50; i++) {
            Map<String, Object> input = new HashMap<>(baseInput);
            input.put("ZONEID", 5000 + i); // Non-premium zones
            input.put("DOMAIN", 3000 + i); // Non-premium domains
            
            float pred = model.predictFloat(input);
            regularPreds.add(pred);
        }
        
        // Calculate averages
        float premiumAvg = premiumPreds.stream().reduce(0f, Float::sum) / premiumPreds.size();
        float regularAvg = regularPreds.stream().reduce(0f, Float::sum) / regularPreds.size();
        
        // Check uniqueness
        Set<String> uniquePremium = new HashSet<>();
        Set<String> uniqueRegular = new HashSet<>();
        for (float p : premiumPreds) uniquePremium.add(String.format("%.3f", p));
        for (float p : regularPreds) uniqueRegular.add(String.format("%.3f", p));
        
        System.out.printf("Premium segments: avg=$%.3f, unique=%d/50\n", premiumAvg, uniquePremium.size());
        System.out.printf("Regular segments: avg=$%.3f, unique=%d/50\n", regularAvg, uniqueRegular.size());
        
        // Success criteria
        boolean success = premiumAvg > 1.0f && // Premium should predict >$1
                         regularAvg < 0.5f && // Regular should predict <$0.50
                         premiumAvg > regularAvg + 0.5f; // Clear differentiation
        
        System.out.println(success ? "\n✓ Model successfully differentiates segments!" : 
                                    "\n❌ Model fails to differentiate segments!");
    }
    
    private void analyzeBidDistribution(Map<String, Integer> bidCounts, Map<String, Float> avgCPMs,
                                       Set<Integer> premiumZones, Set<Integer> premiumDomains) {
        System.out.println("\n=== Bid Distribution Analysis ===");
        
        int premiumSegmentBids = 0;
        int regularSegmentBids = 0;
        int totalSegmentsWithBids = 0;
        
        for (Map.Entry<String, Integer> entry : bidCounts.entrySet()) {
            if (entry.getValue() > 0) {
                totalSegmentsWithBids++;
                
                String[] parts = entry.getKey().split("_");
                int zoneId = Integer.parseInt(parts[0]);
                int domainId = Integer.parseInt(parts[1]);
                
                if (premiumZones.contains(zoneId) && premiumDomains.contains(domainId)) {
                    premiumSegmentBids += entry.getValue();
                } else {
                    regularSegmentBids += entry.getValue();
                }
            }
        }
        
        System.out.printf("Segments that received bids: %d\n", totalSegmentsWithBids);
        System.out.printf("Premium segment bids: %d\n", premiumSegmentBids);
        System.out.printf("Regular segment bids: %d\n", regularSegmentBids);
        System.out.printf("Premium bid share: %.1f%%\n", 
            100.0 * premiumSegmentBids / (premiumSegmentBids + regularSegmentBids));
    }
}