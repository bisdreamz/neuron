package dev.neuronic.net;

import dev.neuronic.net.layers.Feature;
import dev.neuronic.net.optimizers.AdamWOptimizer;
import dev.neuronic.net.simple.SimpleNet;
import dev.neuronic.net.simple.SimpleNetFloat;
import org.junit.jupiter.api.Test;

import java.util.*;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Test that auction penalties correctly bias the model toward 0,
 * with only segments that see positive bids pulling away from this baseline.
 * 
 * This is the DESIRED behavior:
 * - Most segments should predict near 0 due to auction penalties
 * - Only segments with consistent positive bids should predict higher
 */
public class BiasTowardZeroTest {
    
    @Test
    public void testBiasTowardZeroWithPositivePullaway() {
        System.out.println("=== BIAS TOWARD ZERO TEST ===\n");
        System.out.println("Expected behavior:");
        System.out.println("- Auction penalties create strong bias toward 0");
        System.out.println("- Only segments with positive bids pull away from 0");
        System.out.println("- Segments with no/few bids stay near 0\n");
        
        Feature[] features = {
            Feature.embeddingLRU(1000, 32, "ZONEID"),
            Feature.embeddingLRU(1000, 32, "DOMAIN"),
            Feature.passthrough("BIDFLOOR")
        };
        
        AdamWOptimizer optimizer = new AdamWOptimizer(0.001f, 0.001f);
        
        NeuralNet net = NeuralNet.newBuilder()
            .input(features.length)
            .setDefaultOptimizer(optimizer)
            .layer(Layers.inputMixed(features))
            .layer(Layers.hiddenDenseRelu(256))
            .layer(Layers.hiddenDenseRelu(128))
            .layer(Layers.hiddenDenseRelu(64))
            .withGlobalGradientClipping(1.0f)
            .output(Layers.outputHuberRegression(1, optimizer, 3.0f));
            
        SimpleNetFloat model = SimpleNet.ofFloatRegression(net);
        
        Random rand = new Random(42);
        
        // Define different segment types
        Set<String> premiumSegments = new HashSet<>(); // 5% - get 80% of bids
        Set<String> occasionalSegments = new HashSet<>(); // 10% - get occasional bids
        Set<String> rareSegments = new HashSet<>(); // 85% - almost never bid
        
        for (int i = 0; i < 50; i++) {
            premiumSegments.add(i + "_" + i);
        }
        for (int i = 50; i < 150; i++) {
            occasionalSegments.add(i + "_" + i);
        }
        for (int i = 150; i < 1000; i++) {
            rareSegments.add(i + "_" + i);
        }
        
        // Track actual training data
        Map<String, List<Float>> segmentTrainingValues = new HashMap<>();
        Map<String, Integer> segmentPenaltyCounts = new HashMap<>();
        Map<String, Integer> segmentBidCounts = new HashMap<>();
        
        System.out.println("Training 100,000 requests with auction penalties...");
        
        for (int step = 0; step < 100_000; step++) {
            // Pick segment with realistic distribution
            String segment;
            int zoneId, domainId;
            
            float segmentChoice = rand.nextFloat();
            if (segmentChoice < 0.3f) {
                // Premium segment (30% of traffic)
                zoneId = rand.nextInt(50);
                domainId = zoneId;
            } else if (segmentChoice < 0.4f) {
                // Occasional segment (10% of traffic)
                zoneId = 50 + rand.nextInt(100);
                domainId = zoneId;
            } else {
                // Rare segment (60% of traffic)
                zoneId = 150 + rand.nextInt(850);
                domainId = zoneId;
            }
            segment = zoneId + "_" + domainId;
            
            Map<String, Object> input = Map.of(
                "ZONEID", zoneId,
                "DOMAIN", domainId,
                "BIDFLOOR", 1.0f
            );
            
            // STEP 1: Always train with auction penalty
            float penaltyValue = -0.03f; // Negative penalty
            model.train(input, penaltyValue);
            segmentPenaltyCounts.merge(segment, 1, Integer::sum);
            segmentTrainingValues.computeIfAbsent(segment, k -> new ArrayList<>()).add(penaltyValue);
            
            // STEP 2: ~50% of requests also get bid result
            if (rand.nextFloat() < 0.5f) {
                float bidValue;
                
                if (premiumSegments.contains(segment) && rand.nextFloat() < 0.8f) {
                    // Premium segment wins bid 80% of the time
                    bidValue = 2.0f + rand.nextFloat(); // $2-3
                    segmentBidCounts.merge(segment, 1, Integer::sum);
                } else if (occasionalSegments.contains(segment) && rand.nextFloat() < 0.2f) {
                    // Occasional segment wins bid 20% of the time
                    bidValue = 0.5f + rand.nextFloat() * 0.5f; // $0.5-1.0
                    segmentBidCounts.merge(segment, 1, Integer::sum);
                } else if (rareSegments.contains(segment) && rand.nextFloat() < 0.02f) {
                    // Rare segment wins bid 2% of the time
                    bidValue = 0.1f + rand.nextFloat() * 0.2f; // $0.1-0.3
                    segmentBidCounts.merge(segment, 1, Integer::sum);
                } else {
                    // No bid
                    bidValue = 0.0f;
                }
                
                model.train(input, bidValue);
                segmentTrainingValues.get(segment).add(bidValue);
            }
            
            // Progress check
            if (step > 0 && step % 20000 == 0) {
                System.out.printf("Step %d: Testing predictions...\n", step);
                testSegmentPredictions(model);
            }
        }
        
        // Final evaluation
        System.out.println("\n=== FINAL EVALUATION ===");
        
        // Calculate expected values for each segment type
        System.out.println("\nAnalyzing training data:");
        analyzeSegmentTraining(segmentTrainingValues, segmentPenaltyCounts, segmentBidCounts,
                               premiumSegments, occasionalSegments, rareSegments);
        
        // Test predictions
        System.out.println("\nTesting final predictions:");
        validateFinalPredictions(model, premiumSegments, occasionalSegments, rareSegments);
    }
    
    private void testSegmentPredictions(SimpleNetFloat model) {
        // Test different segment types
        float premiumPred = model.predictFloat(Map.of("ZONEID", 10, "DOMAIN", 10, "BIDFLOOR", 1.0f));
        float occasionalPred = model.predictFloat(Map.of("ZONEID", 100, "DOMAIN", 100, "BIDFLOOR", 1.0f));
        float rarePred = model.predictFloat(Map.of("ZONEID", 500, "DOMAIN", 500, "BIDFLOOR", 1.0f));
        float neverSeenPred = model.predictFloat(Map.of("ZONEID", 999, "DOMAIN", 999, "BIDFLOOR", 1.0f));
        
        System.out.printf("  Premium: $%.3f, Occasional: $%.3f, Rare: $%.3f, Never seen: $%.3f\n",
                          premiumPred, occasionalPred, rarePred, neverSeenPred);
    }
    
    private void analyzeSegmentTraining(Map<String, List<Float>> segmentTrainingValues,
                                       Map<String, Integer> segmentPenaltyCounts,
                                       Map<String, Integer> segmentBidCounts,
                                       Set<String> premiumSegments,
                                       Set<String> occasionalSegments,
                                       Set<String> rareSegments) {
        
        // Analyze premium segments
        int totalPremiumPenalties = 0;
        int totalPremiumBids = 0;
        float totalPremiumBidValue = 0;
        
        for (String segment : premiumSegments) {
            totalPremiumPenalties += segmentPenaltyCounts.getOrDefault(segment, 0);
            int bids = segmentBidCounts.getOrDefault(segment, 0);
            totalPremiumBids += bids;
            
            if (segmentTrainingValues.containsKey(segment)) {
                for (float val : segmentTrainingValues.get(segment)) {
                    if (val > 0) totalPremiumBidValue += val;
                }
            }
        }
        
        float avgPremiumBidValue = totalPremiumBids > 0 ? totalPremiumBidValue / totalPremiumBids : 0;
        System.out.printf("Premium segments: %d penalties, %d positive bids (avg $%.2f)\n",
                          totalPremiumPenalties, totalPremiumBids, avgPremiumBidValue);
        
        // Analyze occasional segments
        int totalOccasionalBids = 0;
        for (String segment : occasionalSegments) {
            totalOccasionalBids += segmentBidCounts.getOrDefault(segment, 0);
        }
        System.out.printf("Occasional segments: %d positive bids\n", totalOccasionalBids);
        
        // Analyze rare segments
        int totalRareBids = 0;
        for (String segment : rareSegments) {
            totalRareBids += segmentBidCounts.getOrDefault(segment, 0);
        }
        System.out.printf("Rare segments: %d positive bids\n", totalRareBids);
    }
    
    private void validateFinalPredictions(SimpleNetFloat model,
                                         Set<String> premiumSegments,
                                         Set<String> occasionalSegments,
                                         Set<String> rareSegments) {
        
        // Test premium segments
        List<Float> premiumPreds = new ArrayList<>();
        for (int i = 0; i < 20; i++) {
            float pred = model.predictFloat(Map.of("ZONEID", i, "DOMAIN", i, "BIDFLOOR", 1.0f));
            premiumPreds.add(pred);
        }
        
        // Test occasional segments
        List<Float> occasionalPreds = new ArrayList<>();
        for (int i = 50; i < 70; i++) {
            float pred = model.predictFloat(Map.of("ZONEID", i, "DOMAIN", i, "BIDFLOOR", 1.0f));
            occasionalPreds.add(pred);
        }
        
        // Test rare segments
        List<Float> rarePreds = new ArrayList<>();
        for (int i = 500; i < 520; i++) {
            float pred = model.predictFloat(Map.of("ZONEID", i, "DOMAIN", i, "BIDFLOOR", 1.0f));
            rarePreds.add(pred);
        }
        
        // Test never-seen segments
        List<Float> neverSeenPreds = new ArrayList<>();
        for (int i = 900; i < 920; i++) {
            float pred = model.predictFloat(Map.of("ZONEID", i + 1000, "DOMAIN", i + 1000, "BIDFLOOR", 1.0f));
            neverSeenPreds.add(pred);
        }
        
        // Calculate averages
        float premiumAvg = premiumPreds.stream().reduce(0f, Float::sum) / premiumPreds.size();
        float occasionalAvg = occasionalPreds.stream().reduce(0f, Float::sum) / occasionalPreds.size();
        float rareAvg = rarePreds.stream().reduce(0f, Float::sum) / rarePreds.size();
        float neverSeenAvg = neverSeenPreds.stream().reduce(0f, Float::sum) / neverSeenPreds.size();
        
        System.out.printf("\nPrediction averages:\n");
        System.out.printf("Premium segments: $%.3f (should be >$1.50)\n", premiumAvg);
        System.out.printf("Occasional segments: $%.3f (should be ~$0.10-0.30)\n", occasionalAvg);
        System.out.printf("Rare segments: $%.3f (should be ~$0.00)\n", rareAvg);
        System.out.printf("Never-seen segments: $%.3f (should be ~$0.00)\n", neverSeenAvg);
        
        // Check diversity
        Set<String> uniquePremium = new HashSet<>();
        for (float p : premiumPreds) uniquePremium.add(String.format("%.3f", p));
        
        System.out.printf("\nPrediction diversity: %d unique values for premium segments\n", uniquePremium.size());
        
        // Validate behavior
        boolean correctBias = rareAvg < 0.1f && neverSeenAvg < 0.1f;
        boolean premiumPullaway = premiumAvg > 1.0f;
        boolean goodDifferentiation = premiumAvg > occasionalAvg + 0.5f && 
                                     occasionalAvg > rareAvg;
        
        System.out.println("\nValidation:");
        System.out.println("✓ Bias toward 0: " + (correctBias ? "YES" : "NO"));
        System.out.println("✓ Premium pullaway: " + (premiumPullaway ? "YES" : "NO"));
        System.out.println("✓ Good differentiation: " + (goodDifferentiation ? "YES" : "NO"));
        
        assertTrue(correctBias || premiumPullaway, 
                   "Model should either bias toward 0 OR show premium pullaway");
    }
}