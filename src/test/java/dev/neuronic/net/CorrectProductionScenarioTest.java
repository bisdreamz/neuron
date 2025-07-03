package dev.neuronic.net;

import dev.neuronic.net.layers.Feature;
import dev.neuronic.net.optimizers.AdamWOptimizer;
import dev.neuronic.net.optimizers.Optimizer;
import dev.neuronic.net.optimizers.SgdOptimizer;
import dev.neuronic.net.simple.SimpleNet;
import dev.neuronic.net.simple.SimpleNetFloat;
import org.junit.jupiter.api.Test;

import java.util.*;

import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * CORRECT production scenario matching what you actually do:
 * 
 * 1. EVERY REQUEST gets penalty training (auction cost)
 * 2. ~50% of requests ALSO get bid training (some win, some get 0)
 * 3. Premium segments (5%) get MORE traffic AND higher bid rates
 * 4. Overall: 100,000 penalty trainings, ~50,000 bid trainings
 * 
 * IMPORTANT: IT HAS BEEN PROVEN THAT THE PENALTY IS NOT THE CAUSE OF COLLAPSE!
 * - Test fails even when penalty training is completely removed
 * - Network still collapses to single value without any penalty
 * - The issue is in the neural network library, NOT the training pattern
 * DO NOT INVESTIGATE PENALTY AS A CAUSE - IT IS NOT THE ISSUE!
 */
public class CorrectProductionScenarioTest {
    
    // Toggle penalty training on/off
    private static final boolean ENABLE_PENALTY_TRAINING = true; // Set to false to disable penalty
    
    // Segment configuration
    private static final int numPremiumSegments = 100;  // 100 premium segments
    private static final int numRegularSegments = 5000; // 5000 regular segments
    private final Set<String> premiumSegments = new HashSet<>();
    private final Set<String> regularSegments = new HashSet<>();
    
    // Tracking maps
    private final Map<String, Integer> segmentPenaltyCounts = new HashMap<>();
    private final Map<String, Integer> segmentBidCounts = new HashMap<>();
    private final Map<String, Float> segmentTotalBids = new HashMap<>();
    private final Map<String, Float> segmentCPM = new HashMap<>();
    
    // New tracking for true averages
    private final Map<String, Integer> segmentTotalTrainingCounts = new HashMap<>();
    private final Map<String, Float> segmentTotalTrainingValues = new HashMap<>();
    
    @Test
    public void testCorrectProductionScenario() {
        System.out.println("=== CORRECT PRODUCTION SCENARIO ===\n");
        System.out.println("What actually happens:");
        System.out.println("1. EVERY request → penalty training");
        System.out.println("2. ~50% of requests → ALSO bid training");
        System.out.println("3. Premium segments get MORE traffic");
        System.out.println("3. Premium segments get MORE traffic");
        System.out.println("4. Overall ratio: 2:1 penalties:bids\n");

        int batchSize = 64; // Single sample training like production
        
        // Test with different penalty approaches and batch sizes
        testWithPenalty("penalty (-$0.0003)", 0.05f, batchSize);
        //testWithPenalty("Tiny positive ($0.0001)", 0.0001f, batchSize);
        //testWithPenalty("Very tiny positive ($0.00001)", 0.00001f, batchSize);
    }
    
    private void testWithPenalty(String name, float penaltyValue, int batchSize) {
        System.out.printf("\n=== %s ===\n", name);
        
        // Clear tracking maps for new test
        segmentPenaltyCounts.clear();
        segmentBidCounts.clear();
        segmentTotalBids.clear();
        segmentCPM.clear();
        premiumSegments.clear();
        regularSegments.clear();
        segmentTotalTrainingCounts.clear();
        segmentTotalTrainingValues.clear();
        
        // Initialize segments with CPM values
        Random initRand = new Random(42);
        initializeSegmentCPMs(initRand);
        
        System.out.printf("Batch size: %d, Penalty value: $%.5f\n", batchSize, penaltyValue);

        // Using smaller vocabulary for one-hot to make the test runnable
        int oneHotVocabSize = 1000;

        Feature[] features = {
            Feature.embedding(4, 1,"OS"), // Only using 4 OS values in the test
            Feature.embedding(10000, 64, "ZONEID"),
            Feature.embedding(5000, 32, "DOMAIN"),
            Feature.embedding(2000, 16, "PUB")
            //Feature.autoScale(0f, 20f, "BIDFLOOR")
        };

        //Optimizer optimizer = new SgdOptimizer(0.0002f); // Same as updated SerialCorrectProductionScenarioTest
        //Optimizer optimizer = new AdamWOptimizer(0.0001f, 0f);
        //Optimizer optimizer = new AdamWOptimizer(0.0005f, 0.000001f);
        Optimizer optimizer = new AdamWOptimizer(0.0005f, 0.7f, 0.999f, 1e-8f, 0.000001f);
        int steps = 300_000;  // Match SerialCorrectProductionScenarioTest

        NeuralNet net = NeuralNet.newBuilder()
                .setDefaultOptimizer(optimizer)
                .withGlobalGradientClipping(0f)
                .layer(Layers.inputMixed(features))
                .layer(Layers.hiddenDenseLeakyRelu(64))
                .layer(Layers.hiddenDenseLeakyRelu(32))
                //.output(Layers.outputLinearRegression(1));
                .output(Layers.outputHuberRegression(1, optimizer, 3f));

        SimpleNetFloat model = SimpleNet.ofFloatRegression(net);

        Random rand = new Random(42);

        // Track actual CPM values by feature combination
        Map<String, List<Float>> featureComboCPMs = new HashMap<>();
        Map<String, List<Float>> cpmByOS = new HashMap<>();

        // Tracking
        int totalPenaltyTrains = 0;
        int totalBidTrains = 0;
        
        // Histogram tracking for CPM values
        Map<String, Integer> cpmHistogram = new TreeMap<>();
        
        // Batch accumulation
        List<Map<String, Object>> batchInputs = new ArrayList<>();
        List<Float> batchTargets = new ArrayList<>();

        System.out.println("Simulating " + steps + " requests (with smaller one-hot vocab)...");

        for (int step = 0; step < steps; step++) {
            // Weighted selection: 30% chance for premium segments, 70% for regular
            String segment;
            int zoneId, domainId;
            boolean isPremium = rand.nextFloat() < 0.3f;

            if (isPremium) {
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

            String os = "os_" + rand.nextInt(4);
            Map<String, Object> input = Map.of(
                "OS", os,
                "ZONEID", "zone_" + zoneId,
                "DOMAIN", "domain_" + domainId,
                "PUB", "pub_" + rand.nextInt(2000) // Random publishers
                //"BIDFLOOR", 0.5f + rand.nextFloat()
            );
            
            // STEP 1: Conditionally train with penalty (auction cost)
            if (ENABLE_PENALTY_TRAINING) {
                batchInputs.add(input);
                batchTargets.add(penaltyValue);
                totalPenaltyTrains++;
                segmentPenaltyCounts.merge(segment, 1, Integer::sum);
                
                // Track for true average calculation
                segmentTotalTrainingCounts.merge(segment, 1, Integer::sum);
                segmentTotalTrainingValues.merge(segment, penaltyValue, Float::sum);
            }
            
            // STEP 2: ~50% of requests also get bid training
            if (rand.nextFloat() < 0.5f) {
                float bidRate = isPremium ? 0.8f : 0.05f;
                float bidValue;
                
                if (rand.nextFloat() < bidRate) {
                    // Win bid - use pre-assigned CPM with some noise
                    float baseCPM = segmentCPM.get(segment);
                    float noise = (rand.nextFloat() - 0.5f) * 0.4f; // +/- 20% variation
                    bidValue = Math.max(0.0f, baseCPM + noise);
                } else {
                    // No bid
                    bidValue = 0.0f;
                }
                
                // Add bid training to batch
                // Clone input if we already added penalty, otherwise use original
                batchInputs.add(ENABLE_PENALTY_TRAINING ? new HashMap<>(input) : input);
                batchTargets.add(bidValue);
                totalBidTrains++;
                segmentBidCounts.merge(segment, 1, Integer::sum);
                segmentTotalBids.merge(segment, bidValue, Float::sum);
                
                // Track for true average calculation
                segmentTotalTrainingCounts.merge(segment, 1, Integer::sum);
                segmentTotalTrainingValues.merge(segment, bidValue, Float::sum);
                
                // Track CPM distribution
                String bucket = String.format("$%.1f-%.1f", Math.floor(bidValue), Math.floor(bidValue) + 1);
                cpmHistogram.merge(bucket, 1, Integer::sum);
                
                // Track CPM by OS
                cpmByOS.computeIfAbsent(os, k -> new ArrayList<>()).add(bidValue);
            }
            
            // Train batch when full or at end of data
            if (batchInputs.size() >= batchSize || step == steps - 1) {
                if (!batchInputs.isEmpty()) {
                    // Train the batch using SimpleNetFloat's batch training
                    model.trainBatchMaps(batchInputs, batchTargets);
                    
                    // Clear batch
                    batchInputs.clear();
                    batchTargets.clear();
                }
            }
            
            // Progress check
            if (step > 0 && step % 1000 == 0) {
                System.out.printf("Step %d: %d penalties, %d bids (ratio %.1f:1)\n",
                    step, totalPenaltyTrains, totalBidTrains,
                    (float)totalPenaltyTrains / totalBidTrains);
                
                // Test some predictions
//                float premiumPred = model.predictFloat(Map.of(
//                    "OS", 0, "ZONEID", 10, "DOMAIN", 10, "PUB", 0, "BIDFLOOR", 1.0f));
//                float regularPred = model.predictFloat(Map.of(
//                    "OS", 0, "ZONEID", 5000, "DOMAIN", 1000, "PUB", 0, "BIDFLOOR", 1.0f));
                float premiumPred = model.predictFloat(Map.of(
                        "OS", "os_0", "ZONEID", "zone_10", "DOMAIN", "domain_10", "PUB", "pub_10"));
                float regularPred = model.predictFloat(Map.of(
                        "OS", "os_0", "ZONEID", "zone_5000", "DOMAIN", "domain_0", "PUB", "pub_1500"));
                
                System.out.printf("  Premium: $%.3f, Regular: $%.3f\n", premiumPred, regularPred);
            }
        }
        
        // Final statistics
        System.out.printf("\nFinal: %d penalty trains, %d bid trains", 
            totalPenaltyTrains, totalBidTrains);
        if (totalBidTrains > 0) {
            System.out.printf(" (ratio %.1f:1)", (float)totalPenaltyTrains / totalBidTrains);
        }
        System.out.println();
        
        // Print CPM histogram
        System.out.println("\nCPM Distribution Histogram:");
        for (Map.Entry<String, Integer> entry : cpmHistogram.entrySet()) {
            int count = entry.getValue();
            System.out.printf("%s: %4d [", entry.getKey(), count);
            for (int i = 0; i < Math.min(50, count / 10); i++) System.out.print("█");
            System.out.println("]");
        }
        
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
        
        // Show CPM distribution by OS
        System.out.println("\nCPM Distribution by OS:");
        for (int i = 0; i < 4; i++) {
            String osKey = "os_" + i;
            List<Float> cpms = cpmByOS.get(osKey);
            if (cpms != null && !cpms.isEmpty()) {
                float avgCPM = cpms.stream().reduce(0f, Float::sum) / cpms.size();
                float maxCPM = cpms.stream().max(Float::compare).orElse(0f);
                System.out.printf("  %s: avg=$%.2f, max=$%.2f, count=%d\n", 
                    osKey, avgCPM, maxCPM, cpms.size());
            }
        }
        
        // Show penalty vs CPM contribution analysis
        System.out.println("\nValue Contribution Analysis:");
        analyzeValueContributions(penaltyValue);
        
        // Print sample segment CPMs to verify data generation
        System.out.println("\nSample segment CPMs (first 10 premium, 10 regular):");
        int count = 0;
        for (Map.Entry<String, Float> entry : segmentCPM.entrySet()) {
            if (count < 10 && premiumSegments.contains(entry.getKey())) {
                System.out.printf("  Premium %s: $%.2f\n", entry.getKey(), entry.getValue());
                count++;
            }
        }
        count = 0;
        for (Map.Entry<String, Float> entry : segmentCPM.entrySet()) {
            if (count < 10 && regularSegments.contains(entry.getKey())) {
                System.out.printf("  Regular %s: $%.2f\n", entry.getKey(), entry.getValue());
                count++;
            }
        }
        
        // Test final predictions
        System.out.println("\nTesting prediction accuracy:");
        evaluatePredictions(model, segmentCPM, segmentBidCounts, segmentTotalBids, penaltyValue);
        
        // Show detailed histogram analysis
        evaluatePredictionsWithHistogram(model, penaltyValue);
    }
    
    private void evaluatePredictions(SimpleNetFloat model, Map<String, Float> segmentCPM,
                                    Map<String, Integer> segmentBidCounts,
                                    Map<String, Float> segmentTotalBids, float penaltyValue) {
        List<Float> premiumPreds = new ArrayList<>();
        List<Float> premiumExpected = new ArrayList<>();
        List<Float> regularPreds = new ArrayList<>();
        List<Float> regularExpected = new ArrayList<>();
        
        // Test premium segments
        for (int i = 0; i < 50; i++) {
            String segment = i + "_" + i;
            float pred = model.predictFloat(Map.of(
                "OS", "os_0", "ZONEID", "zone_" + i, "DOMAIN", "domain_" + i, "PUB", "pub_" + i));
            premiumPreds.add(pred);
            
            // Calculate expected based on all training values (penalty + CPM)
            if (segmentTotalTrainingCounts.containsKey(segment)) {
                float avgTrainingValue = segmentTotalTrainingValues.get(segment) / segmentTotalTrainingCounts.get(segment);
                premiumExpected.add(avgTrainingValue);
            }
        }
        
        // Test regular segments
        for (int i = 5000; i < 5050; i++) {
            int domainId = i % 5000;
            String segment = i + "_" + domainId;
            float pred = model.predictFloat(Map.of(
                "OS", "os_0", "ZONEID", "zone_" + i, "DOMAIN", "domain_" + domainId, "PUB", "pub_" + (1000 + (i % 1000))));
            regularPreds.add(pred);
            
            if (segmentTotalTrainingCounts.containsKey(segment)) {
                float avgTrainingValue = segmentTotalTrainingValues.get(segment) / segmentTotalTrainingCounts.get(segment);
                regularExpected.add(avgTrainingValue);
            }
        }
        
        // Calculate averages
        float premiumAvgPred = premiumPreds.stream().reduce(0f, Float::sum) / premiumPreds.size();
        float regularAvgPred = regularPreds.stream().reduce(0f, Float::sum) / regularPreds.size();
        
        // Calculate expected averages from actual training data
        float premiumAvgExpected = 0;
        float regularAvgExpected = 0;
        
        if (!premiumExpected.isEmpty()) {
            premiumAvgExpected = premiumExpected.stream().reduce(0f, Float::sum) / premiumExpected.size();
        } else {
            // Estimate based on typical premium patterns
            float typicalPremiumCPM = 3.5f;
            float premiumBidRate = 0.8f;
            premiumAvgExpected = penaltyValue + (premiumBidRate * typicalPremiumCPM);
        }
        
        if (!regularExpected.isEmpty()) {
            regularAvgExpected = regularExpected.stream().reduce(0f, Float::sum) / regularExpected.size();
        } else {
            // Estimate based on typical regular patterns
            float typicalRegularCPM = 0.3f;
            float regularBidRate = 0.05f;
            regularAvgExpected = penaltyValue + (regularBidRate * typicalRegularCPM);
        }
        
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
        assertTrue(success, String.format("Model failed to differentiate segments. Premium avg: $%.3f, Regular avg: $%.3f, Unique values: %d", 
                   premiumAvgPred, regularAvgPred, uniquePremium.size()));
    }
    
    private record PredictionResult(float premiumPred, float regularPred, int step) {}
    
    private void analyzeValueContributions(float penaltyValue) {
        // Analyze a few sample segments
        int premiumSamples = 0;
        float totalPremiumExpected = 0;
        float totalPremiumCPMContribution = 0;
        
        for (String segment : premiumSegments) {
            if (segmentTotalTrainingCounts.containsKey(segment) && premiumSamples < 20) {
                float avgTrainingValue = segmentTotalTrainingValues.get(segment) / segmentTotalTrainingCounts.get(segment);
                int penaltyCount = segmentPenaltyCounts.getOrDefault(segment, 0);
                int bidCount = segmentBidCounts.getOrDefault(segment, 0);
                float avgBidValue = bidCount > 0 ? segmentTotalBids.get(segment) / bidCount : 0;
                
                // Calculate contributions
                float penaltyContribution = penaltyValue * penaltyCount / (float)(penaltyCount + bidCount);
                float cpmContribution = avgBidValue * bidCount / (float)(penaltyCount + bidCount);
                
                totalPremiumExpected += avgTrainingValue;
                totalPremiumCPMContribution += cpmContribution;
                premiumSamples++;
            }
        }
        
        int regularSamples = 0;
        float totalRegularExpected = 0;
        float totalRegularCPMContribution = 0;
        
        for (String segment : regularSegments) {
            if (segmentTotalTrainingCounts.containsKey(segment) && regularSamples < 20) {
                float avgTrainingValue = segmentTotalTrainingValues.get(segment) / segmentTotalTrainingCounts.get(segment);
                int penaltyCount = segmentPenaltyCounts.getOrDefault(segment, 0);
                int bidCount = segmentBidCounts.getOrDefault(segment, 0);
                float avgBidValue = bidCount > 0 ? segmentTotalBids.get(segment) / bidCount : 0;
                
                // Calculate contributions
                float penaltyContribution = penaltyValue * penaltyCount / (float)(penaltyCount + bidCount);
                float cpmContribution = avgBidValue * bidCount / (float)(penaltyCount + bidCount);
                
                totalRegularExpected += avgTrainingValue;
                totalRegularCPMContribution += cpmContribution;
                regularSamples++;
            }
        }
        
        if (premiumSamples > 0) {
            float avgPremiumExpected = totalPremiumExpected / premiumSamples;
            float avgPremiumCPM = totalPremiumCPMContribution / premiumSamples;
            System.out.printf("  Premium segments: Expected avg=$%.3f (penalty=$%.3f + CPM contribution=$%.3f)\n",
                avgPremiumExpected, penaltyValue, avgPremiumCPM);
        }
        
        if (regularSamples > 0) {
            float avgRegularExpected = totalRegularExpected / regularSamples;
            float avgRegularCPM = totalRegularCPMContribution / regularSamples;
            System.out.printf("  Regular segments: Expected avg=$%.3f (penalty=$%.3f + CPM contribution=$%.3f)\n",
                avgRegularExpected, penaltyValue, avgRegularCPM);
        }
    }
    
    private void initializeSegmentCPMs(Random rand) {
        // Initialize premium segments with high CPMs distributed across features
        for (int i = 0; i < numPremiumSegments; i++) {
            String segment = i + "_" + i;
            premiumSegments.add(segment);
            
            // Base CPM $2-5, but vary by index to ensure distribution
            float baseCPM = 2.0f + (i % 4) * 0.75f; // Distributes across 2.0, 2.75, 3.5, 4.25
            float randomVariation = rand.nextFloat() * 1.0f; // Add 0-1.0 random
            segmentCPM.put(segment, baseCPM + randomVariation);
        }
        
        // Initialize regular segments with low CPMs
        for (int i = 0; i < numRegularSegments; i++) {
            int zoneId = numPremiumSegments + i;
            int domainId = zoneId % 5000;
            String segment = zoneId + "_" + domainId;
            regularSegments.add(segment);
            
            // Base CPM $0.1-0.5
            float baseCPM = 0.1f + rand.nextFloat() * 0.4f;
            segmentCPM.put(segment, baseCPM);
        }
        
        System.out.printf("Initialized %d premium segments (CPM $2-5) and %d regular segments (CPM $0.1-0.5)\n",
            premiumSegments.size(), regularSegments.size());
    }
    
    private void evaluatePredictionsWithHistogram(SimpleNetFloat model, float penaltyValue) {
        System.out.println("\n=== ACTUAL vs PREDICTED VALUE ANALYSIS ===");
        
        List<Float> allActuals = new ArrayList<>();
        List<Float> allPredictions = new ArrayList<>();
        List<Float> percentageErrors = new ArrayList<>();
        
        // Test a sample of segments
        Random testRand = new Random(123);
        int samplesPerType = 100;
        
        // Test premium segments
        List<String> premiumList = new ArrayList<>(premiumSegments);
        Collections.shuffle(premiumList, testRand);
        for (int i = 0; i < Math.min(samplesPerType, premiumList.size()); i++) {
            String segment = premiumList.get(i);
            String[] parts = segment.split("_");
            int zoneId = Integer.parseInt(parts[0]);
            int domainId = Integer.parseInt(parts[1]);
            
            // Test with different OS values to ensure distribution
            for (int os = 0; os < 4; os++) {
                Map<String, Object> input = Map.of(
                    "OS", "os_" + os,
                    "ZONEID", "zone_" + zoneId,
                    "DOMAIN", "domain_" + domainId,
                    "PUB", "pub_" + (zoneId % 2000)
                );
                
                // Get the true expected value (average of all training values)
                float actual;
                if (segmentTotalTrainingCounts.containsKey(segment)) {
                    actual = segmentTotalTrainingValues.get(segment) / segmentTotalTrainingCounts.get(segment);
                } else {
                    // If not trained, use CPM + penalty estimate
                    actual = penaltyValue + (0.8f * segmentCPM.get(segment));
                }
                
                float predicted = model.predictFloat(input);
                
                allActuals.add(actual);
                allPredictions.add(predicted);
                
                // Calculate percentage error with safeguards
                float error;
                if (Math.abs(actual) < 0.01f) {
                    // For very small values, use absolute error instead
                    error = Math.abs(predicted - actual) * 100;
                } else {
                    error = (predicted - actual) / Math.abs(actual) * 100;
                    // Cap extreme percentage errors
                    error = Math.max(-200f, Math.min(200f, error));
                }
                percentageErrors.add(error);
            }
        }
        
        // Test regular segments
        List<String> regularList = new ArrayList<>(regularSegments);
        Collections.shuffle(regularList, testRand);
        for (int i = 0; i < Math.min(samplesPerType, regularList.size()); i++) {
            String segment = regularList.get(i);
            String[] parts = segment.split("_");
            int zoneId = Integer.parseInt(parts[0]);
            int domainId = Integer.parseInt(parts[1]);
            
            Map<String, Object> input = Map.of(
                "OS", "os_" + testRand.nextInt(4),
                "ZONEID", "zone_" + zoneId,
                "DOMAIN", "domain_" + domainId,
                "PUB", "pub_" + (zoneId % 2000)
            );
            
            // Get the true expected value (average of all training values)
            float actual;
            if (segmentTotalTrainingCounts.containsKey(segment)) {
                actual = segmentTotalTrainingValues.get(segment) / segmentTotalTrainingCounts.get(segment);
            } else {
                // If not trained, use CPM + penalty estimate
                actual = penaltyValue + (0.05f * segmentCPM.get(segment));
            }
            
            float predicted = model.predictFloat(input);
            
            allActuals.add(actual);
            allPredictions.add(predicted);
            
            float error;
            if (Math.abs(actual) < 0.01f) {
                // For very small values, use absolute error instead
                error = Math.abs(predicted - actual) * 100;
            } else {
                error = (predicted - actual) / Math.abs(actual) * 100;
                // Cap extreme percentage errors
                error = Math.max(-200f, Math.min(200f, error));
            }
            percentageErrors.add(error);
        }
        
        // Create histogram buckets for actual vs predicted
        Map<String, Integer> actualHistogram = new TreeMap<>();
        Map<String, Integer> predictedHistogram = new TreeMap<>();
        
        for (int i = 0; i < allActuals.size(); i++) {
            String actualBucket = String.format("$%.1f-%.1f", 
                Math.floor(allActuals.get(i)), Math.floor(allActuals.get(i)) + 1);
            String predictedBucket = String.format("$%.1f-%.1f", 
                Math.floor(allPredictions.get(i)), Math.floor(allPredictions.get(i)) + 1);
            
            actualHistogram.merge(actualBucket, 1, Integer::sum);
            predictedHistogram.merge(predictedBucket, 1, Integer::sum);
        }
        
        // Print side-by-side histogram
        System.out.println("\nActual vs Predicted Value Distribution:");
        System.out.println("Value Range | Actual Count | Predicted Count");
        System.out.println("----------|--------------|----------------");
        
        Set<String> allBuckets = new TreeSet<>();
        allBuckets.addAll(actualHistogram.keySet());
        allBuckets.addAll(predictedHistogram.keySet());
        
        for (String bucket : allBuckets) {
            int actualCount = actualHistogram.getOrDefault(bucket, 0);
            int predictedCount = predictedHistogram.getOrDefault(bucket, 0);
            System.out.printf("%-9s | %12d | %15d\n", bucket, actualCount, predictedCount);
        }
        
        // Percentage error histogram
        Map<String, Integer> errorHistogram = new TreeMap<>();
        for (float error : percentageErrors) {
            String bucket;
            if (error < -50) bucket = "< -50%";
            else if (error < -25) bucket = "-50% to -25%";
            else if (error < -10) bucket = "-25% to -10%";
            else if (error < 0) bucket = "-10% to 0%";
            else if (error < 10) bucket = "0% to +10%";
            else if (error < 25) bucket = "+10% to +25%";
            else if (error < 50) bucket = "+25% to +50%";
            else bucket = "> +50%";
            
            errorHistogram.merge(bucket, 1, Integer::sum);
        }
        
        System.out.println("\nPrediction Error Distribution:");
        System.out.println("Error Range    | Count | Visual");
        System.out.println("---------------|-------|--------");
        for (Map.Entry<String, Integer> entry : errorHistogram.entrySet()) {
            int count = entry.getValue();
            System.out.printf("%-14s | %5d | ", entry.getKey(), count);
            for (int i = 0; i < Math.min(50, count / 10); i++) System.out.print("█");
            System.out.println();
        }
        
        // Calculate statistics
        float meanAbsoluteError = 0;
        float meanAbsolutePercentageError = 0;
        float rmse = 0;
        
        for (int i = 0; i < allActuals.size(); i++) {
            float error = allPredictions.get(i) - allActuals.get(i);
            meanAbsoluteError += Math.abs(error);
            meanAbsolutePercentageError += Math.abs(percentageErrors.get(i));
            rmse += error * error;
        }
        
        meanAbsoluteError /= allActuals.size();
        meanAbsolutePercentageError /= allActuals.size();
        rmse = (float) Math.sqrt(rmse / allActuals.size());
        
        System.out.printf("\nError Statistics:\n");
        System.out.printf("  Mean Absolute Error (MAE): $%.4f\n", meanAbsoluteError);
        System.out.printf("  Mean Absolute Percentage Error (MAPE): %.2f%%\n", meanAbsolutePercentageError);
        System.out.printf("  Root Mean Square Error (RMSE): $%.4f\n", rmse);
        
        // Check if model is learning properly - separate by known segment type
        float avgPremiumActual = 0, avgPremiumPred = 0;
        float avgRegularActual = 0, avgRegularPred = 0;
        int premiumCount = 0, regularCount = 0;
        
        // We know first samplesPerType*4 predictions are premium (4 OS values each)
        int premiumSamples = Math.min(samplesPerType, premiumSegments.size()) * 4;
        
        for (int i = 0; i < allActuals.size(); i++) {
            if (i < premiumSamples) { // Premium segments
                avgPremiumActual += allActuals.get(i);
                avgPremiumPred += allPredictions.get(i);
                premiumCount++;
            } else { // Regular segments
                avgRegularActual += allActuals.get(i);
                avgRegularPred += allPredictions.get(i);
                regularCount++;
            }
        }
        
        // Safe division with checks
        if (premiumCount > 0) {
            avgPremiumActual /= premiumCount;
            avgPremiumPred /= premiumCount;
        }
        if (regularCount > 0) {
            avgRegularActual /= regularCount;
            avgRegularPred /= regularCount;
        }
        
        System.out.printf("\nSegment Analysis:\n");
        if (premiumCount > 0) {
            float premiumError = (Math.abs(avgPremiumActual) > 0.01f) ? 
                (avgPremiumPred - avgPremiumActual) / avgPremiumActual * 100 : 0;
            System.out.printf("  Premium (%d samples) - Actual: $%.3f, Predicted: $%.3f (%.1f%% error)\n", 
                premiumCount, avgPremiumActual, avgPremiumPred, premiumError);
        } else {
            System.out.println("  Premium - No samples found");
        }
        
        if (regularCount > 0) {
            float regularError = (Math.abs(avgRegularActual) > 0.01f) ? 
                (avgRegularPred - avgRegularActual) / avgRegularActual * 100 : 0;
            System.out.printf("  Regular (%d samples) - Actual: $%.3f, Predicted: $%.3f (%.1f%% error)\n", 
                regularCount, avgRegularActual, avgRegularPred, regularError);
        } else {
            System.out.println("  Regular - No samples found");
        }
        
        boolean success = meanAbsolutePercentageError < 25.0f && 
                         avgPremiumPred > avgRegularPred * 2;
        
        System.out.println(success ? "\n[SUCCESS] Model learns CPM patterns accurately!" : 
                                    "\n[FAILURE] Model struggles to learn CPM patterns!");
        assertTrue(success, String.format("Model failed to learn CPM patterns. MAPE: %.2f%%, Premium avg: $%.3f, Regular avg: $%.3f", 
                   meanAbsolutePercentageError, avgPremiumPred, avgRegularPred));
    }
}