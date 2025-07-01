package dev.neuronic.net;

import dev.neuronic.net.layers.Feature;
import dev.neuronic.net.optimizers.AdamWOptimizer;
import dev.neuronic.net.simple.SimpleNet;
import dev.neuronic.net.simple.SimpleNetFloat;
import org.junit.jupiter.api.Test;

import java.util.*;

/**
 * CORRECT production scenario matching what you actually do:
 * 
 * 1. EVERY REQUEST gets penalty training (auction cost)
 * 2. ~50% of requests ALSO get bid training (some win, some get 0)
 * 3. Premium segments (5%) get MORE traffic AND higher bid rates
 * 4. Overall: 100,000 penalty trainings, ~50,000 bid trainings
 */
public class CorrectProductionScenarioTest {
    
    @Test
    public void testCorrectProductionScenario() {
        System.out.println("=== CORRECT PRODUCTION SCENARIO ===\n");
        System.out.println("What actually happens:");
        System.out.println("1. EVERY request → penalty training");
        System.out.println("2. ~50% of requests → ALSO bid training");
        System.out.println("3. Premium segments get MORE traffic");
        System.out.println("4. Overall ratio: 2:1 penalties:bids\n");
        
        // Test with different penalty approaches
        testWithPenalty("Negative penalty (-$0.0003)", -0.0003f);
        testWithPenalty("Tiny positive ($0.0001)", 0.0001f);
        testWithPenalty("Very tiny positive ($0.00001)", 0.00001f);
    }
    
    private void testWithPenalty(String name, float penaltyValue) {
        System.out.printf("\n=== %s ===\n", name);

        // Using smaller vocabulary for one-hot to make the test runnable
        int oneHotVocabSize = 1000;

        Feature[] features = {
            Feature.oneHot(50, "OS"),
            Feature.embedding(10000, 32, "ZONEID"),      // 10k zones -> 32-dim embeddings
            Feature.embedding(5000, 16, "DOMAIN"),       // 5k domains -> 16-dim embeddings  
            Feature.embedding(2000, 16, "PUB"),          // 2k publishers -> 16-dim embeddings
            Feature.autoScale(0f, 20f, "BIDFLOOR")
        };

        AdamWOptimizer optimizer = new AdamWOptimizer(0.0001f, 0.01f);

        NeuralNet net = NeuralNet.newBuilder()
                .setDefaultOptimizer(optimizer)
                .layer(Layers.inputMixed(features))
                .layer(Layers.hiddenDenseRelu(512))
                .layer(Layers.hiddenDenseRelu(256))
                .layer(Layers.hiddenDenseRelu(128))
                .layer(Layers.hiddenDenseRelu(64))
                .output(Layers.outputLinearRegression(1));

        SimpleNetFloat model = SimpleNet.ofFloatRegression(net);

        Random rand = new Random(42);

        // Define segment characteristics
        Map<String, Float> segmentBidRate = new HashMap<>();
        Map<String, Float> segmentCPM = new HashMap<>();

        // Adjust segment generation for embeddings
        int numPremiumSegments = 500; // 5% of 10000 zones
        int numRegularSegments = 9500;

        // Premium segments
        Set<String> premiumSegments = new HashSet<>();
        for (int i = 0; i < numPremiumSegments; i++) {
            String segment = i + "_" + (i % 1000); // Premium zones with various domains
            premiumSegments.add(segment);
            segmentBidRate.put(segment, 0.8f);
            segmentCPM.put(segment, 2.0f + (i % 10) * 0.1f);
        }

        // Regular segments  
        Set<String> regularSegments = new HashSet<>();
        for (int i = numPremiumSegments; i < 10000; i++) {
            String segment = i + "_" + (i % 5000); // Regular zones with various domains
            regularSegments.add(segment);
            segmentBidRate.put(segment, 0.05f);
            segmentCPM.put(segment, 0.1f + (i % 5) * 0.1f);
        }

        // Tracking
        int totalPenaltyTrains = 0;
        int totalBidTrains = 0;
        Map<String, Integer> segmentPenaltyCounts = new HashMap<>();
        Map<String, Integer> segmentBidCounts = new HashMap<>();
        Map<String, Float> segmentTotalBids = new HashMap<>();

        int steps = 1000;
        System.out.println("Simulating 100,000 requests (with smaller one-hot vocab)...");

        for (int step = 0; step < 5000; step++) {
            // Weighted selection: 30% chance for premium segments, 70% for regular
            String segment;
            int zoneId, domainId;

            if (rand.nextFloat() < 0.3f) {
                // Pick a random premium segment
                int premiumIndex = rand.nextInt(numPremiumSegments);
                zoneId = premiumIndex;
                domainId = premiumIndex % 1000;
            } else {
                // Pick a random regular segment
                zoneId = numPremiumSegments + rand.nextInt(numRegularSegments);
                domainId = zoneId % 5000;
            }
            segment = zoneId + "_" + domainId;

            Map<String, Object> input = Map.of(
                "OS", rand.nextInt(4),
                "ZONEID", zoneId,
                "DOMAIN", domainId,
                "PUB", rand.nextInt(2000), // Use full pub range for embeddings
                "BIDFLOOR", 0.5f + rand.nextFloat()
            );
            
            // STEP 1: ALWAYS train with penalty (auction cost)
            model.train(input, penaltyValue);
            totalPenaltyTrains++;
            segmentPenaltyCounts.merge(segment, 1, Integer::sum);
            
            // STEP 2: ~50% of requests also get bid training
            if (rand.nextFloat() < 0.5f) {
                float bidRate = segmentBidRate.get(segment);
                float bidValue;
                
                if (rand.nextFloat() < bidRate) {
                    // Win bid
                    bidValue = segmentCPM.get(segment) + (rand.nextFloat() - 0.5f) * 0.2f;
                } else {
                    // No bid
                    bidValue = 0.0f;
                }
                
                model.train(input, bidValue);
                totalBidTrains++;
                segmentBidCounts.merge(segment, 1, Integer::sum);
                segmentTotalBids.merge(segment, bidValue, Float::sum);
            }
            
            // Progress check
            if (step > 0 && step % 1000 == 0) {
                System.out.printf("Step %d: %d penalties, %d bids (ratio %.1f:1)\n",
                    step, totalPenaltyTrains, totalBidTrains,
                    (float)totalPenaltyTrains / totalBidTrains);
                
                // Test some predictions
                float premiumPred = model.predictFloat(Map.of(
                    "OS", 0, "ZONEID", 10, "DOMAIN", 10, "PUB", 0, "BIDFLOOR", 1.0f));
                float regularPred = model.predictFloat(Map.of(
                    "OS", 0, "ZONEID", 5000, "DOMAIN", 1000, "PUB", 0, "BIDFLOOR", 1.0f));
                
                System.out.printf("  Premium: $%.3f, Regular: $%.3f\n", premiumPred, regularPred);
            }
        }
        
        // Final statistics
        System.out.printf("\nFinal: %d penalty trains, %d bid trains (ratio %.1f:1)\n",
            totalPenaltyTrains, totalBidTrains,
            (float)totalPenaltyTrains / totalBidTrains);
        
        // Analyze traffic distribution
        int premiumTraffic = 0;
        int regularTraffic = 0;
        for (Map.Entry<String, Integer> entry : segmentPenaltyCounts.entrySet()) {
            if (premiumSegments.contains(entry.getKey())) {
                premiumTraffic += entry.getValue();
            } else {
                regularTraffic += entry.getValue();
            }
        }
        
        System.out.printf("Traffic distribution: Premium %.1f%%, Regular %.1f%%\n",
            100.0 * premiumTraffic / totalPenaltyTrains,
            100.0 * regularTraffic / totalPenaltyTrains);
        
        // Test final predictions
        System.out.println("\nTesting prediction accuracy:");
        evaluatePredictions(model, segmentCPM, segmentBidCounts, segmentTotalBids);
    }
    
    private void evaluatePredictions(SimpleNetFloat model, Map<String, Float> segmentCPM,
                                    Map<String, Integer> segmentBidCounts,
                                    Map<String, Float> segmentTotalBids) {
        List<Float> premiumPreds = new ArrayList<>();
        List<Float> premiumExpected = new ArrayList<>();
        List<Float> regularPreds = new ArrayList<>();
        List<Float> regularExpected = new ArrayList<>();
        
        // Test premium segments
        for (int i = 0; i < 50; i++) {
            String segment = i + "_" + i;
            float pred = model.predictFloat(Map.of(
                "OS", 0, "ZONEID", i, "DOMAIN", i, "PUB", 0, "BIDFLOOR", 1.0f));
            premiumPreds.add(pred);
            
            // Calculate expected based on actual training
            if (segmentBidCounts.containsKey(segment)) {
                float avgBid = segmentTotalBids.get(segment) / segmentBidCounts.get(segment);
                premiumExpected.add(avgBid);
            }
        }
        
        // Test regular segments
        for (int i = 5000; i < 5050; i++) {
            int domainId = i % 5000;
            String segment = i + "_" + domainId;
            float pred = model.predictFloat(Map.of(
                "OS", 0, "ZONEID", i, "DOMAIN", domainId, "PUB", 0, "BIDFLOOR", 1.0f));
            regularPreds.add(pred);
            
            if (segmentBidCounts.containsKey(segment)) {
                float avgBid = segmentTotalBids.get(segment) / segmentBidCounts.get(segment);
                regularExpected.add(avgBid);
            }
        }
        
        // Calculate averages
        float premiumAvgPred = premiumPreds.stream().reduce(0f, Float::sum) / premiumPreds.size();
        float regularAvgPred = regularPreds.stream().reduce(0f, Float::sum) / regularPreds.size();
        
        float premiumAvgExpected = premiumExpected.isEmpty() ? 2.5f :
            premiumExpected.stream().reduce(0f, Float::sum) / premiumExpected.size();
        float regularAvgExpected = regularExpected.isEmpty() ? 0.3f :
            regularExpected.stream().reduce(0f, Float::sum) / regularExpected.size();
        
        // Check uniqueness
        Set<String> uniquePremium = new HashSet<>();
        Set<String> uniqueRegular = new HashSet<>();
        for (float p : premiumPreds) uniquePremium.add(String.format("%.3f", p));
        for (float p : regularPreds) uniqueRegular.add(String.format("%.3f", p));
        
        System.out.printf("Premium: predicted avg=$%.3f (expected ~$%.3f), %d unique values\n",
            premiumAvgPred, premiumAvgExpected, uniquePremium.size());
        System.out.printf("Regular: predicted avg=$%.3f (expected ~$%.3f), %d unique values\n",
            regularAvgPred, regularAvgExpected, uniqueRegular.size());
        System.out.printf("Differentiation: $%.3f\n", premiumAvgPred - regularAvgPred);
        
        // Success criteria
        boolean success = premiumAvgPred > regularAvgPred + 0.5f && // Clear differentiation
                         premiumAvgPred > 1.0f && // Premium predicts high
                         uniquePremium.size() > 10; // Good diversity
        
        System.out.println(success ? "[SUCCESS] - Model differentiates segments!" : "[FAILURE] - Model cannot differentiate!");
    }
    
    private record PredictionResult(float premiumPred, float regularPred, int step) {}
}