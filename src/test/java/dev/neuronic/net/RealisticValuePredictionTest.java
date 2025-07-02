package dev.neuronic.net;

import dev.neuronic.net.layers.Feature;
import dev.neuronic.net.optimizers.AdamWOptimizer;
import dev.neuronic.net.optimizers.Optimizer;
import dev.neuronic.net.simple.SimpleNet;
import dev.neuronic.net.simple.SimpleNetFloat;
import org.junit.jupiter.api.Test;

import java.util.*;

/**
 * Realistic test that models predicting EXPECTED VALUE (bid_rate × CPM) 
 * rather than just CPM. This matches production use case where:
 * 
 * 1. Segments have varying bid rates (0.25% - 3%) AND varying CPMs
 * 2. Expected value = (bid_rate × avg_CPM) - auction_cost
 * 3. Only a sample of auctions train with penalty (downsampling)
 * 4. All winning bids train with actual CPM
 * 
 * The model must learn that:
 * - High CPM + Low bid rate can be worse than Low CPM + High bid rate
 * - Some segments have negative expected value (cost > revenue)
 */
public class RealisticValuePredictionTest {
    
    // Realistic bid rates in the 0.25% - 3% range
    private static final float MIN_BID_RATE = 0.0025f;  // 0.25%
    private static final float MAX_BID_RATE = 0.03f;    // 3%
    
    // Penalty sampling rate (like your 5% in production)
    private static final float PENALTY_SAMPLE_RATE = 0.5f;
    
    // Auction cost (penalty)
    private static final float AUCTION_COST = -0.001f;
    
    @Test
    public void testExpectedValueLearning() {
        System.out.println("=== REALISTIC EXPECTED VALUE PREDICTION TEST ===\n");
        System.out.println("Testing neural network learning of expected value per request");
        System.out.println("Expected Value = (Bid Rate × CPM) - Auction Cost\n");
        
        // Create diverse segments with varying bid rates and CPMs
        int numSegments = 1000;
        Map<String, SegmentProfile> segments = createSegmentProfiles(numSegments);
        
        // Model configuration matching production
        Feature[] features = {
            Feature.embedding(4, 4, "OS"),
            Feature.embedding(numSegments, 32, "SEGMENT"),
            Feature.embedding(100, 16, "PUB"),
            Feature.autoScale(0f, 20f, "BIDFLOOR")
        };
        
        Optimizer optimizer = new AdamWOptimizer(0.001f, 0.000001f);
        
        NeuralNet net = NeuralNet.newBuilder()
                .setDefaultOptimizer(optimizer)
                .withGlobalGradientClipping(10f)
                .layer(Layers.inputMixed(features))
                .layer(Layers.hiddenDenseRelu(64))
                .layer(Layers.hiddenDenseRelu(32))
                .output(Layers.outputHuberRegression(1, optimizer, 3f));
        
        SimpleNetFloat model = SimpleNet.ofFloatRegression(net);
        
        // Training
        Random rand = new Random(42);
        int totalSteps = 500_000;
        
        // Tracking
        int penaltyTrains = 0;
        int bidTrains = 0;
        Map<String, Float> segmentRevenue = new HashMap<>();
        Map<String, Integer> segmentAuctions = new HashMap<>();
        
        System.out.println("Training with realistic bid patterns...\n");
        
        for (int step = 0; step < totalSteps; step++) {
            // Pick a random segment
            String segmentId = "seg_" + rand.nextInt(numSegments);
            SegmentProfile profile = segments.get(segmentId);
            
            Map<String, Object> input = Map.of(
                "OS", "os_" + rand.nextInt(4),
                "SEGMENT", segmentId,
                "PUB", "pub_" + rand.nextInt(100),
                "BIDFLOOR", 0.5f + rand.nextFloat() * 2f
            );
            
            segmentAuctions.merge(segmentId, 1, Integer::sum);
            
            // ALWAYS an auction happens (cost incurred)
            // But only sample some for penalty training
            if (rand.nextFloat() < PENALTY_SAMPLE_RATE) {
                model.train(input, -AUCTION_COST);
                penaltyTrains++;
            }
            
            // Check if this auction gets a bid
            if (rand.nextFloat() < profile.bidRate) {
                // Generate CPM with some variance
                float cpm = profile.generateCPM(rand);
                model.train(input, cpm);
                bidTrains++;
                
                segmentRevenue.merge(segmentId, cpm - AUCTION_COST, Float::sum);
            } else {
                // No bid = just cost
                segmentRevenue.merge(segmentId, -AUCTION_COST, Float::sum);
            }
            
            // Progress update
            if (step > 0 && step % 10000 == 0) {
                System.out.printf("Step %d: %d penalty trains, %d bid trains (sample rate %.1f%%)\n",
                    step, penaltyTrains, bidTrains, 100f * penaltyTrains / step);
                
                // Test some predictions
                evaluateSegments(model, segments, segmentRevenue, segmentAuctions, 5);
            }
        }
        
        // Final evaluation
        System.out.println("\n=== FINAL EVALUATION ===");
        evaluateAllSegments(model, segments, segmentRevenue, segmentAuctions);
    }
    
    private Map<String, SegmentProfile> createSegmentProfiles(int numSegments) {
        Map<String, SegmentProfile> segments = new HashMap<>();
        Random rand = new Random(123);
        
        for (int i = 0; i < numSegments; i++) {
            String segmentId = "seg_" + i;
            
            // Create diverse segments with uncorrelated bid rates and CPMs
            float bidRate = MIN_BID_RATE + rand.nextFloat() * (MAX_BID_RATE - MIN_BID_RATE);
            
            // CPM distribution independent of bid rate
            float avgCPM;
            float cpmVariance;
            
            float segmentType = rand.nextFloat();
            if (segmentType < 0.1f) {
                // 10% high-value segments
                avgCPM = 2f + rand.nextFloat() * 3f;  // $2-5
                cpmVariance = 0.5f;
            } else if (segmentType < 0.3f) {
                // 20% medium-value segments  
                avgCPM = 0.5f + rand.nextFloat() * 1.5f;  // $0.50-2
                cpmVariance = 0.3f;
            } else {
                // 70% low-value segments
                avgCPM = 0.05f + rand.nextFloat() * 0.45f;  // $0.05-0.50
                cpmVariance = 0.1f;
            }
            
            segments.put(segmentId, new SegmentProfile(bidRate, avgCPM, cpmVariance));
        }
        
        // Log some example segments
        System.out.println("Sample segment profiles:");
        segments.entrySet().stream()
                .limit(10)
                .forEach(e -> {
                    SegmentProfile p = e.getValue();
                    float expectedValue = (p.bidRate * p.avgCPM) - AUCTION_COST;
                    System.out.printf("  %s: bid_rate=%.2f%%, avg_CPM=$%.2f, expected_value=$%.4f\n",
                        e.getKey(), p.bidRate * 100, p.avgCPM, expectedValue);
                });
        System.out.println();
        
        return segments;
    }
    
    private void evaluateSegments(SimpleNetFloat model, Map<String, SegmentProfile> segments,
                                  Map<String, Float> segmentRevenue, Map<String, Integer> segmentAuctions,
                                  int numSamples) {
        List<String> sampleSegments = segments.keySet().stream()
                .sorted()
                .limit(numSamples)
                .toList();
        
        for (String segmentId : sampleSegments) {
            float prediction = model.predictFloat(Map.of(
                "OS", "os_0",
                "SEGMENT", segmentId,
                "PUB", "pub_0",
                "BIDFLOOR", 1.0f
            ));
            
            SegmentProfile profile = segments.get(segmentId);
            float expectedValue = (profile.bidRate * profile.avgCPM) - AUCTION_COST;
            
            System.out.printf("  %s: predicted=$%.4f, expected=$%.4f (bid_rate=%.1f%%, cpm=$%.2f)\n",
                segmentId, prediction, expectedValue, profile.bidRate * 100, profile.avgCPM);
        }
    }
    
    private void evaluateAllSegments(SimpleNetFloat model, Map<String, SegmentProfile> segments,
                                     Map<String, Float> segmentRevenue, Map<String, Integer> segmentAuctions) {
        List<Float> predictions = new ArrayList<>();
        List<Float> expectedValues = new ArrayList<>();
        List<Float> actualRevenues = new ArrayList<>();
        
        // Categories for analysis
        int positiveValueCount = 0;
        int negativeValueCount = 0;
        float posPredSum = 0, posExpectedSum = 0;
        float negPredSum = 0, negExpectedSum = 0;
        
        for (Map.Entry<String, SegmentProfile> entry : segments.entrySet()) {
            String segmentId = entry.getKey();
            SegmentProfile profile = entry.getValue();
            
            // Get prediction
            float prediction = model.predictFloat(Map.of(
                "OS", "os_0",
                "SEGMENT", segmentId,
                "PUB", "pub_0",
                "BIDFLOOR", 1.0f
            ));
            
            // Calculate theoretical expected value
            float expectedValue = (profile.bidRate * profile.avgCPM) - AUCTION_COST;
            
            predictions.add(prediction);
            expectedValues.add(expectedValue);
            
            // Track actual revenue if we have data
            if (segmentAuctions.containsKey(segmentId)) {
                float actualRevPerAuction = segmentRevenue.getOrDefault(segmentId, 0f) / 
                                           segmentAuctions.get(segmentId);
                actualRevenues.add(actualRevPerAuction);
            }
            
            // Categorize
            if (expectedValue > 0) {
                positiveValueCount++;
                posPredSum += prediction;
                posExpectedSum += expectedValue;
            } else {
                negativeValueCount++;
                negPredSum += prediction;
                negExpectedSum += expectedValue;
            }
        }
        
        // Calculate metrics
        float mae = 0;
        float rmse = 0;
        for (int i = 0; i < predictions.size(); i++) {
            float error = predictions.get(i) - expectedValues.get(i);
            mae += Math.abs(error);
            rmse += error * error;
        }
        mae /= predictions.size();
        rmse = (float) Math.sqrt(rmse / predictions.size());
        
        // Print results
        System.out.printf("\nSegment Distribution:\n");
        System.out.printf("  Positive value segments: %d (%.1f%%)\n", 
            positiveValueCount, 100f * positiveValueCount / segments.size());
        System.out.printf("  Negative value segments: %d (%.1f%%)\n", 
            negativeValueCount, 100f * negativeValueCount / segments.size());
        
        System.out.printf("\nPrediction Accuracy:\n");
        System.out.printf("  Positive segments - Avg predicted: $%.4f, Avg expected: $%.4f\n",
            posPredSum / positiveValueCount, posExpectedSum / positiveValueCount);
        System.out.printf("  Negative segments - Avg predicted: $%.4f, Avg expected: $%.4f\n",
            negPredSum / negativeValueCount, negExpectedSum / negativeValueCount);
        
        System.out.printf("\nError Metrics:\n");
        System.out.printf("  Mean Absolute Error: $%.4f\n", mae);
        System.out.printf("  Root Mean Square Error: $%.4f\n", rmse);
        
        // Success criteria
        boolean success = mae < 0.005f && // Good accuracy
                         (posPredSum / positiveValueCount) > 0 && // Positive segments predicted positive
                         (negPredSum / negativeValueCount) < 0;   // Negative segments predicted negative
        
        System.out.println(success ? "\n[SUCCESS] Model learns expected value correctly!" : 
                                    "\n[FAILURE] Model struggles with expected value!");
        
        // Show some interesting segments
        System.out.println("\nInteresting segments (high bid rate + low CPM vs low bid rate + high CPM):");
        segments.entrySet().stream()
                .sorted((a, b) -> {
                    float aValue = (a.getValue().bidRate * a.getValue().avgCPM) - AUCTION_COST;
                    float bValue = (b.getValue().bidRate * b.getValue().avgCPM) - AUCTION_COST;
                    return Float.compare(bValue, aValue);
                })
                .limit(5)
                .forEach(e -> {
                    SegmentProfile p = e.getValue();
                    float pred = model.predictFloat(Map.of(
                        "OS", "os_0", "SEGMENT", e.getKey(), "PUB", "pub_0", "BIDFLOOR", 1.0f));
                    float expected = (p.bidRate * p.avgCPM) - AUCTION_COST;
                    System.out.printf("  %s: bid_rate=%.2f%%, cpm=$%.2f → expected=$%.4f, predicted=$%.4f\n",
                        e.getKey(), p.bidRate * 100, p.avgCPM, expected, pred);
                });
    }
    
    // Helper class to define segment characteristics
    private static class SegmentProfile {
        final float bidRate;
        final float avgCPM;
        final float cpmVariance;
        
        SegmentProfile(float bidRate, float avgCPM, float cpmVariance) {
            this.bidRate = bidRate;
            this.avgCPM = avgCPM;
            this.cpmVariance = cpmVariance;
        }
        
        float generateCPM(Random rand) {
            // Generate CPM with normal distribution around average
            float cpm = (float) (avgCPM + (rand.nextGaussian() * cpmVariance));
            return Math.max(0.01f, cpm); // Minimum CPM
        }
    }
}