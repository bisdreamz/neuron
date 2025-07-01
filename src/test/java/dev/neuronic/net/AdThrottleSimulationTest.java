package dev.neuronic.net;

import dev.neuronic.net.layers.Feature;
import dev.neuronic.net.optimizers.AdamWOptimizer;
import dev.neuronic.net.simple.SimpleNet;
import dev.neuronic.net.simple.SimpleNetFloat;
import org.junit.jupiter.api.Test;

import java.util.*;
import java.util.concurrent.ThreadLocalRandom;
import java.util.stream.Collectors;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Comprehensive test simulating real-world ad throttling with skewed distributions.
 * Models the scenario where 5% of inventory gets 80% of traffic (power law distribution).
 */
public class AdThrottleSimulationTest {
    
    // Inventory sizes matching real-world scale
    private static final int NUM_APPS = 1000;
    private static final int NUM_ZONES = 4000;
    private static final int NUM_PUBLISHERS = 100;
    private static final int NUM_COUNTRIES = 20;
    private static final int NUM_OS = 4; // ios, android, roku, other
    private static final int NUM_DEVICE_TYPES = 5; // phone, tablet, tv, desktop, other
    
    // Training parameters
    private static final int TOTAL_REQUESTS = 100_000;
    private static final float BID_RATE = 0.015f; // 1.5% bid rate
    private static final float AUCTION_SAMPLE_RATE = 0.02f; // 2% of auctions trained
    
    // Value distributions
    private static final float MIN_CPM = 0.10f;
    private static final float MAX_CPM = 3.0f;
    private static final float NO_BID_VALUE = -0.5f; // Very strong negative signal
    
    private final Random rand = new Random(42); // Reproducible
    
    // Premium inventory (top 5% that gets 80% of traffic)
    private Set<String> premiumApps;
    private Set<String> premiumZones;
    private Set<Integer> premiumPublishers;
    
    // CPM rates by segment
    private Map<String, Float> appBaseCpm;
    private Map<String, Float> countryMultiplier;
    private Map<String, Float> osMultiplier;
    
    @Test
    public void testOnlineLearning() {
        System.out.println("=== ONLINE LEARNING TEST (Train as you go) ===");
        
        // Initialize inventory with power law distribution
        initializeInventory();
        
        // Create neural network
        SimpleNetFloat model = createModel();
        
        // Simulate online training with realistic traffic patterns
        TrainingStats stats = simulateOnlineTraining(model);
        
        // Evaluate model performance
        evaluateModel(model, stats);
    }
    
    @Test
    public void testBulkTraining() {
        System.out.println("=== BULK TRAINING TEST (Collect then train) ===");
        
        // Initialize inventory with power law distribution
        initializeInventory();
        
        // Create neural network
        SimpleNetFloat model = createModel();
        
        // Collect training data first
        List<TrainingExample> trainingData = collectTrainingData();
        
        // Bulk train the model
        TrainingStats stats = bulkTrain(model, trainingData);
        
        // Evaluate model performance on test set
        evaluateModel(model, stats);
    }
    
    @Test 
    public void testCompareApproaches() {
        System.out.println("=== COMPARING ONLINE VS BULK TRAINING ===");
        
        initializeInventory();
        
        // Online learning
        SimpleNetFloat onlineModel = createModel();
        TrainingStats onlineStats = simulateOnlineTraining(onlineModel);
        
        // Bulk training
        SimpleNetFloat bulkModel = createModel();
        List<TrainingExample> trainingData = collectTrainingData();
        TrainingStats bulkStats = bulkTrain(bulkModel, trainingData);
        
        // Compare results
        System.out.println("\n=== COMPARISON ===");
        System.out.println("Online Learning:");
        float onlineScore = evaluateModelScore(onlineModel);
        System.out.println("\nBulk Training:");
        float bulkScore = evaluateModelScore(bulkModel);
        
        System.out.printf("\nOnline score: %.3f\n", onlineScore);
        System.out.printf("Bulk score: %.3f\n", bulkScore);
    }
    
    private void initializeInventory() {
        // Select top 5% as premium
        premiumApps = new HashSet<>();
        premiumZones = new HashSet<>();
        premiumPublishers = new HashSet<>();
        
        for (int i = 0; i < NUM_APPS * 0.05; i++) {
            premiumApps.add("app_" + i);
        }
        
        for (int i = 0; i < NUM_ZONES * 0.05; i++) {
            premiumZones.add("zone_" + i);
        }
        
        for (int i = 0; i < NUM_PUBLISHERS * 0.05; i++) {
            premiumPublishers.add(i);
        }
        
        // Initialize CPM values
        appBaseCpm = new HashMap<>();
        for (int i = 0; i < NUM_APPS; i++) {
            String app = "app_" + i;
            if (premiumApps.contains(app)) {
                // Premium apps: $0.50 - $3.00
                appBaseCpm.put(app, 0.5f + rand.nextFloat() * 2.5f);
            } else {
                // Regular apps: $0.10 - $0.50
                appBaseCpm.put(app, 0.1f + rand.nextFloat() * 0.4f);
            }
        }
        
        // Country multipliers (US/UK/CA premium)
        countryMultiplier = new HashMap<>();
        countryMultiplier.put("US", 1.0f);
        countryMultiplier.put("UK", 0.9f);
        countryMultiplier.put("CA", 0.85f);
        countryMultiplier.put("AU", 0.8f);
        countryMultiplier.put("DE", 0.7f);
        countryMultiplier.put("FR", 0.65f);
        countryMultiplier.put("JP", 0.6f);
        countryMultiplier.put("BR", 0.3f);
        countryMultiplier.put("IN", 0.2f);
        countryMultiplier.put("MX", 0.25f);
        // Fill rest with lower values
        for (int i = 10; i < NUM_COUNTRIES; i++) {
            countryMultiplier.put("C" + i, 0.1f + rand.nextFloat() * 0.2f);
        }
        
        // OS multipliers
        osMultiplier = new HashMap<>();
        osMultiplier.put("ios", 1.2f);
        osMultiplier.put("android", 0.8f);
        osMultiplier.put("roku", 0.5f);
        osMultiplier.put("other", 0.3f);
    }
    
    private SimpleNetFloat createModel() {
        Feature[] features = {
            Feature.oneHot(25, "country"), // Extra space for dictionary growth
            Feature.oneHot(NUM_OS, "os"),
            Feature.oneHot(NUM_DEVICE_TYPES, "device_type"),
            Feature.embeddingLRU(NUM_PUBLISHERS, 8, "publisher"),
            Feature.embeddingLRU(NUM_APPS, 32, "app"),
            Feature.embeddingLRU(NUM_ZONES, 16, "zone"),
            Feature.autoScale(0f, 5f, "bid_floor")
        };
        
        AdamWOptimizer optimizer = new AdamWOptimizer(0.01f, 0.001f);
        
        NeuralNet net = NeuralNet.newBuilder()
            .input(features.length)
            .withGlobalGradientClipping(1.0f)
            .setDefaultOptimizer(optimizer)
            .layer(Layers.inputMixed(features))
            .layer(Layers.hiddenDenseRelu(128))
            .layer(Layers.dropout(0.2f))
            .layer(Layers.hiddenDenseRelu(64))
            .output(Layers.outputLinearRegression(1));
            
        return SimpleNet.ofFloatRegression(net);
    }
    
    private Map<String, Object> generateRequest() {
        Map<String, Object> request = new HashMap<>();
        
        // Power law distribution - 80% chance of premium inventory
        boolean isPremium = rand.nextFloat() < 0.8f;
        
        // Select app with skewed distribution
        String app;
        if (isPremium && !premiumApps.isEmpty()) {
            app = premiumApps.toArray(new String[0])[rand.nextInt(premiumApps.size())];
        } else {
            app = "app_" + (50 + rand.nextInt(NUM_APPS - 50));
        }
        
        // Select zone with skewed distribution
        String zone;
        if (isPremium && !premiumZones.isEmpty()) {
            zone = premiumZones.toArray(new String[0])[rand.nextInt(premiumZones.size())];
        } else {
            zone = "zone_" + (200 + rand.nextInt(NUM_ZONES - 200));
        }
        
        // Select publisher
        int publisher = isPremium && !premiumPublishers.isEmpty() 
            ? premiumPublishers.iterator().next() 
            : 5 + rand.nextInt(NUM_PUBLISHERS - 5);
            
        // Country distribution (50% US, 20% tier-1, 30% rest)
        String country;
        float countryRoll = rand.nextFloat();
        if (countryRoll < 0.5f) {
            country = "US";
        } else if (countryRoll < 0.7f) {
            country = new String[]{"UK", "CA", "AU", "DE"}[rand.nextInt(4)];
        } else {
            country = "C" + (5 + rand.nextInt(15));
        }
        
        // OS distribution (40% iOS, 50% Android, 10% other)
        String os;
        float osRoll = rand.nextFloat();
        if (osRoll < 0.4f) {
            os = "ios";
        } else if (osRoll < 0.9f) {
            os = "android";
        } else {
            os = rand.nextBoolean() ? "roku" : "other";
        }
        
        request.put("app", app);
        request.put("zone", zone);
        request.put("publisher", publisher);
        request.put("country", country);
        request.put("os", os);
        request.put("device_type", rand.nextInt(NUM_DEVICE_TYPES));
        request.put("bid_floor", 0.01f + rand.nextFloat() * 0.5f);
        
        return request;
    }
    
    private float calculateBidValue(Map<String, Object> request) {
        String app = (String) request.get("app");
        String country = (String) request.get("country");
        String os = (String) request.get("os");
        float bidFloor = (Float) request.get("bid_floor");
        
        // Base CPM from app quality
        float baseCpm = appBaseCpm.getOrDefault(app, 0.2f);
        
        // Apply multipliers
        float countryCpm = baseCpm * countryMultiplier.getOrDefault(country, 0.3f);
        float finalCpm = countryCpm * osMultiplier.getOrDefault(os, 0.5f);
        
        // Add some noise
        finalCpm *= (0.8f + rand.nextFloat() * 0.4f);
        
        // Respect bid floor
        finalCpm = Math.max(finalCpm, bidFloor);
        
        // Cap at reasonable max
        return Math.min(finalCpm, MAX_CPM);
    }
    
    private static class TrainingExample {
        final Map<String, Object> features;
        final float label;
        final boolean isBid;
        
        TrainingExample(Map<String, Object> features, float label, boolean isBid) {
            this.features = features;
            this.label = label;
            this.isBid = isBid;
        }
    }
    
    private List<TrainingExample> collectTrainingData() {
        List<TrainingExample> examples = new ArrayList<>();
        
        for (int i = 0; i < TOTAL_REQUESTS; i++) {
            Map<String, Object> request = generateRequest();
            float trueValue = calculateBidValue(request);
            
            // Decide if we bid
            boolean shouldBid = trueValue >= 0.15f && rand.nextFloat() < BID_RATE;
            
            if (shouldBid) {
                examples.add(new TrainingExample(request, trueValue, true));
            } else if (rand.nextFloat() < AUCTION_SAMPLE_RATE) {
                examples.add(new TrainingExample(request, NO_BID_VALUE, false));
            }
        }
        
        System.out.printf("Collected %d training examples (%d bids, %d no-bids)\n",
            examples.size(),
            examples.stream().filter(e -> e.isBid).count(),
            examples.stream().filter(e -> !e.isBid).count());
            
        return examples;
    }
    
    private TrainingStats bulkTrain(SimpleNetFloat model, List<TrainingExample> examples) {
        TrainingStats stats = new TrainingStats();
        
        // Shuffle for better training
        Collections.shuffle(examples, rand);
        
        // Train in mini-batches
        int batchSize = 32;
        int epochs = 3;
        
        System.out.printf("Bulk training with %d examples, batch size %d, %d epochs\n", 
            examples.size(), batchSize, epochs);
            
        for (int epoch = 0; epoch < epochs; epoch++) {
            float epochLoss = 0;
            
            for (int i = 0; i < examples.size(); i += batchSize) {
                int end = Math.min(i + batchSize, examples.size());
                
                // Train batch
                for (int j = i; j < end; j++) {
                    TrainingExample ex = examples.get(j);
                    model.train(ex.features, ex.label);
                    
                    // Track stats
                    if (ex.isBid) {
                        stats.recordBid(ex.features, ex.label, 0);
                    } else {
                        stats.recordAuction(ex.features, 0);
                    }
                }
                
                // Progress update
                if ((i / batchSize) % 100 == 0) {
                    System.out.printf("  Epoch %d: %d/%d batches\n", 
                        epoch + 1, i / batchSize, examples.size() / batchSize);
                }
            }
        }
        
        stats.total = examples.size() * epochs;
        return stats;
    }
    
    private float evaluateModelScore(SimpleNetFloat model) {
        // Comprehensive scoring based on multiple factors
        float score = 0;
        
        // 1. Premium vs regular discrimination
        float premiumAvg = 0, regularAvg = 0;
        int premiumCount = 0, regularCount = 0;
        
        for (int i = 0; i < 1000; i++) {
            Map<String, Object> request = generateRequest();
            float pred = model.predictFloat(request);
            
            if (premiumApps.contains(request.get("app"))) {
                premiumAvg += pred;
                premiumCount++;
            } else {
                regularAvg += pred;
                regularCount++;
            }
        }
        
        premiumAvg /= premiumCount;
        regularAvg /= regularCount;
        float discrimination = premiumAvg - regularAvg;
        score += Math.max(0, Math.min(1, discrimination * 2)); // 0-1 score
        
        // 2. Prediction diversity
        Set<String> uniquePreds = new HashSet<>();
        for (int i = 0; i < 1000; i++) {
            float pred = model.predictFloat(generateRequest());
            uniquePreds.add(String.format("%.3f", pred));
        }
        float diversity = Math.min(1, uniquePreds.size() / 100.0f);
        score += diversity;
        
        // 3. Correlation with true values
        float correlation = calculateCorrelation(model, 1000);
        score += Math.max(0, correlation);
        
        return score / 3.0f; // Average of three metrics
    }
    
    private float calculateCorrelation(SimpleNetFloat model, int samples) {
        float[] predictions = new float[samples];
        float[] trueValues = new float[samples];
        
        for (int i = 0; i < samples; i++) {
            Map<String, Object> request = generateRequest();
            predictions[i] = model.predictFloat(request);
            trueValues[i] = calculateBidValue(request);
        }
        
        // Simple correlation calculation
        float predMean = 0, trueMean = 0;
        for (int i = 0; i < samples; i++) {
            predMean += predictions[i];
            trueMean += trueValues[i];
        }
        predMean /= samples;
        trueMean /= samples;
        
        float numerator = 0, predVar = 0, trueVar = 0;
        for (int i = 0; i < samples; i++) {
            float predDiff = predictions[i] - predMean;
            float trueDiff = trueValues[i] - trueMean;
            numerator += predDiff * trueDiff;
            predVar += predDiff * predDiff;
            trueVar += trueDiff * trueDiff;
        }
        
        if (predVar == 0 || trueVar == 0) return 0;
        return numerator / (float) Math.sqrt(predVar * trueVar);
    }
    
    private TrainingStats simulateOnlineTraining(SimpleNetFloat model) {
        TrainingStats stats = new TrainingStats();
        float adaptiveThreshold = 0.05f; // Start low
        
        System.out.println("\nTraining with " + TOTAL_REQUESTS + " requests...");
        
        for (int i = 0; i < TOTAL_REQUESTS; i++) {
            Map<String, Object> request = generateRequest();
            
            // Get model prediction
            float prediction = model.predictFloat(request);
            
            // Adaptive threshold logic (mimics your QPS management)
            if (i % 1000 == 0 && i > 0) {
                float passRate = (float) stats.passed / stats.total;
                if (passRate > 0.3f) {
                    adaptiveThreshold *= 1.05f; // Raise threshold if too many pass
                } else if (passRate < 0.1f) {
                    adaptiveThreshold *= 0.95f; // Lower if too few pass
                }
                adaptiveThreshold = Math.max(0.01f, Math.min(2.0f, adaptiveThreshold));
            }
            
            boolean wouldPass = prediction >= adaptiveThreshold;
            
            // Decide if we bid (based on value)
            float trueValue = calculateBidValue(request);
            boolean shouldBid = trueValue >= adaptiveThreshold * 1.2f; // Some margin
            
            // Training logic
            if (shouldBid && rand.nextFloat() < BID_RATE) {
                // Bid - train with actual value
                model.train(request, trueValue);
                stats.recordBid(request, trueValue, prediction);
            } else if (rand.nextFloat() < AUCTION_SAMPLE_RATE) {
                // No bid - train with negative value
                model.train(request, NO_BID_VALUE);
                stats.recordAuction(request, prediction);
            }
            
            stats.total++;
            if (wouldPass) stats.passed++;
            
            // Progress updates
            if (i % 10000 == 0 && i > 0) {
                System.out.printf("Progress: %d requests, %.1f%% pass rate, threshold=%.3f%n",
                    i, (stats.passed * 100.0 / stats.total), adaptiveThreshold);
                
                // Sample predictions
                samplePredictions(model, 5);
            }
        }
        
        return stats;
    }
    
    private void samplePredictions(SimpleNetFloat model, int samples) {
        System.out.println("  Sample predictions:");
        for (int i = 0; i < samples; i++) {
            Map<String, Object> request = generateRequest();
            float prediction = model.predictFloat(request);
            float trueValue = calculateBidValue(request);
            
            System.out.printf("    %s/%s/%s: pred=%.3f, true=%.3f%n",
                request.get("app"), request.get("os"), request.get("country"),
                prediction, trueValue);
        }
    }
    
    private void evaluateModel(SimpleNetFloat model, TrainingStats stats) {
        System.out.println("\n=== MODEL EVALUATION ===");
        
        // Test 1: Premium vs Non-Premium Discrimination
        System.out.println("\n1. Premium vs Non-Premium Apps:");
        float avgPremium = 0, avgRegular = 0;
        int premiumCount = 0, regularCount = 0;
        
        for (int i = 0; i < 1000; i++) {
            Map<String, Object> request = generateRequest();
            float prediction = model.predictFloat(request);
            
            String app = (String) request.get("app");
            if (premiumApps.contains(app)) {
                avgPremium += prediction;
                premiumCount++;
            } else {
                avgRegular += prediction;
                regularCount++;
            }
        }
        
        avgPremium /= premiumCount;
        avgRegular /= regularCount;
        
        System.out.printf("  Premium apps avg: %.3f%n", avgPremium);
        System.out.printf("  Regular apps avg: %.3f%n", avgRegular);
        System.out.printf("  Difference: %.3f (should be positive)%n", avgPremium - avgRegular);
        
        // Test 2: Country Value Recognition
        System.out.println("\n2. Country Value Recognition:");
        for (String country : Arrays.asList("US", "UK", "BR", "IN")) {
            float avgValue = testCountryValue(model, country);
            System.out.printf("  %s average: %.3f%n", country, avgValue);
        }
        
        // Test 3: OS Platform Discrimination
        System.out.println("\n3. OS Platform Values:");
        for (String os : Arrays.asList("ios", "android", "roku")) {
            float avgValue = testOsValue(model, os);
            System.out.printf("  %s average: %.3f%n", os, avgValue);
        }
        
        // Test 4: Prediction Distribution
        System.out.println("\n4. Prediction Distribution:");
        analyzePredictionDistribution(model, 10000);
        
        // Test 5: Learning Quality
        System.out.println("\n5. Learning Quality Metrics:");
        System.out.printf("  Total requests: %d%n", stats.total);
        System.out.printf("  Bids: %d (%.2f%%)%n", stats.bids, 100.0 * stats.bids / stats.total);
        System.out.printf("  Auction samples: %d%n", stats.auctionSamples);
        System.out.printf("  Unique apps seen: %d%n", stats.uniqueApps.size());
        System.out.printf("  Unique zones seen: %d%n", stats.uniqueZones.size());
        
        // Assertions
        assertTrue(avgPremium > avgRegular, 
            "Model should predict higher values for premium apps");
        assertTrue(avgPremium - avgRegular > 0.1f, 
            "Premium/regular difference should be substantial");
    }
    
    private float testCountryValue(SimpleNetFloat model, String country) {
        float sum = 0;
        int count = 100;
        
        for (int i = 0; i < count; i++) {
            Map<String, Object> request = generateRequest();
            request.put("country", country);
            sum += model.predictFloat(request);
        }
        
        return sum / count;
    }
    
    private float testOsValue(SimpleNetFloat model, String os) {
        float sum = 0;
        int count = 100;
        
        for (int i = 0; i < count; i++) {
            Map<String, Object> request = generateRequest();
            request.put("os", os);
            sum += model.predictFloat(request);
        }
        
        return sum / count;
    }
    
    private void analyzePredictionDistribution(SimpleNetFloat model, int samples) {
        List<Float> predictions = new ArrayList<>();
        
        for (int i = 0; i < samples; i++) {
            Map<String, Object> request = generateRequest();
            predictions.add(model.predictFloat(request));
        }
        
        predictions.sort(Float::compareTo);
        
        float min = predictions.get(0);
        float p10 = predictions.get(samples / 10);
        float p50 = predictions.get(samples / 2);
        float p90 = predictions.get(9 * samples / 10);
        float max = predictions.get(samples - 1);
        
        System.out.printf("  Min: %.3f%n", min);
        System.out.printf("  10th percentile: %.3f%n", p10);
        System.out.printf("  Median: %.3f%n", p50);
        System.out.printf("  90th percentile: %.3f%n", p90);
        System.out.printf("  Max: %.3f%n", max);
        System.out.printf("  Range: %.3f%n", max - min);
        
        // Check for mode collapse
        Set<String> uniqueValues = predictions.stream()
            .map(p -> String.format("%.4f", p))
            .collect(Collectors.toSet());
        
        System.out.printf("  Unique values (4 decimals): %d%n", uniqueValues.size());
        
        assertTrue(uniqueValues.size() > 100, 
            "Model should produce diverse predictions, not collapse to few values");
        assertTrue(max - min > 0.5f, 
            "Prediction range should be substantial");
    }
    
    private static class TrainingStats {
        int total = 0;
        int passed = 0;
        int bids = 0;
        int auctionSamples = 0;
        Set<String> uniqueApps = new HashSet<>();
        Set<String> uniqueZones = new HashSet<>();
        
        void recordBid(Map<String, Object> request, float value, float prediction) {
            bids++;
            uniqueApps.add((String) request.get("app"));
            uniqueZones.add((String) request.get("zone"));
        }
        
        void recordAuction(Map<String, Object> request, float prediction) {
            auctionSamples++;
            uniqueApps.add((String) request.get("app"));
            uniqueZones.add((String) request.get("zone"));
        }
    }
}