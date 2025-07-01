package dev.neuronic.net;

import dev.neuronic.net.layers.Feature;
import dev.neuronic.net.optimizers.AdamWOptimizer;
import dev.neuronic.net.simple.SimpleNet;
import dev.neuronic.net.simple.SimpleNetFloat;
import org.junit.jupiter.api.Test;

import java.util.*;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.ThreadLocalRandom;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Test that simulates always-on production environment where the model
 * must continuously learn and predict while maintaining stability.
 * No separate train/test phases - just continuous online operation.
 */
public class AlwaysOnStabilityTest {
    private volatile boolean running = true;
    
    @Test
    public void testAlwaysOnStability() throws InterruptedException {
        System.out.println("=== ALWAYS-ON STABILITY TEST ===\n");
        System.out.println("Simulating continuous production environment:");
        System.out.println("- 5,000 requests/second");
        System.out.println("- 2% training rate (100 trains/second)");
        System.out.println("- Continuous prediction on ALL requests");
        System.out.println("- Multi-threaded concurrent access");
        System.out.println("- No train/test split - pure online learning\n");
        
        // Your production features
        Feature[] features = {
            Feature.oneHot(10, "os"),
            Feature.embeddingLRU(100, 8, "pubid"),
            Feature.hashedEmbedding(10000, 16, "app_bundle"),
            Feature.embeddingLRU(4000, 12, "zone_id"),
            Feature.oneHot(7, "device_type"),
            Feature.oneHot(5, "connection_type"),
            Feature.passthrough("bid_floor")
        };
        
        // Use the configuration that worked
        AdamWOptimizer optimizer = new AdamWOptimizer(0.001f, 0.001f);
        
        NeuralNet net = NeuralNet.newBuilder()
            .input(features.length)
            .setDefaultOptimizer(optimizer)
            .layer(Layers.inputMixed(features))
            .layer(Layers.hiddenDenseRelu(256))
            .layer(Layers.hiddenDenseRelu(128))
            .layer(Layers.hiddenDenseRelu(64))
            .withGlobalGradientClipping(1.0f)
            .output(Layers.outputLinearRegression(1));
            
        SimpleNetFloat model = SimpleNet.ofFloatRegression(net);
        
        // Define segment quality (like production)
        Set<String> premiumApps = new HashSet<>();
        Set<Integer> premiumPubs = new HashSet<>(Arrays.asList(1, 2, 3, 4, 5));
        for (int i = 0; i < 50; i++) {
            premiumApps.add("com.premium.app" + i);
        }
        
        // Metrics tracking
        AtomicLong totalRequests = new AtomicLong(0);
        AtomicLong totalTrains = new AtomicLong(0);
        AtomicLong totalPredictions = new AtomicLong(0);
        AtomicInteger collapseCount = new AtomicInteger(0);
        
        // Use multiple threads to simulate production load
        ExecutorService executor = Executors.newFixedThreadPool(16);
        running = true;
        
        // Start monitoring thread
        Thread monitor = new Thread(() -> {
            long lastRequests = 0;
            long lastTrains = 0;
            long lastPredictions = 0;
            long startTime = System.currentTimeMillis();
            
            while (running) {
                try {
                    Thread.sleep(5000); // Check every 5 seconds
                    
                    long currentRequests = totalRequests.get();
                    long currentTrains = totalTrains.get();
                    long currentPredictions = totalPredictions.get();
                    long elapsedSeconds = (System.currentTimeMillis() - startTime) / 1000;
                    
                    // Calculate rates
                    long requestRate = (currentRequests - lastRequests) / 5;
                    long trainRate = (currentTrains - lastTrains) / 5;
                    long predictRate = (currentPredictions - lastPredictions) / 5;
                    
                    System.out.printf("\n[%ds] Rates: %d req/s, %d train/s, %d pred/s\n",
                        elapsedSeconds, requestRate, trainRate, predictRate);
                    System.out.printf("  Total: %d requests, %d trains (%.1f%%), %d predictions\n",
                        currentRequests, currentTrains, 
                        currentRequests > 0 ? 100.0 * currentTrains / currentRequests : 0,
                        currentPredictions);
                    
                    // Test segment differentiation
                    testStability(model, premiumApps, collapseCount);
                    
                    lastRequests = currentRequests;
                    lastTrains = currentTrains;
                    lastPredictions = currentPredictions;
                    
                } catch (InterruptedException e) {
                    break;
                }
            }
        });
        monitor.start();
        
        // Simulate production traffic for 60 seconds
        long endTime = System.currentTimeMillis() + 60_000;
        
        while (System.currentTimeMillis() < endTime) {
            // Submit batches of requests
            for (int i = 0; i < 100; i++) {
                executor.submit(() -> {
                    ThreadLocalRandom rand = ThreadLocalRandom.current();
                    
                    // Generate request
                    Map<String, Object> input = new HashMap<>();
                    float segmentDraw = rand.nextFloat();
                    float expectedValue;
                    
                    if (segmentDraw < 0.05f) {
                        // Premium segment (5%)
                        input.put("os", "ios");
                        input.put("pubid", premiumPubs.toArray()[rand.nextInt(premiumPubs.size())]);
                        input.put("app_bundle", "com.premium.app" + rand.nextInt(50));
                        input.put("zone_id", rand.nextInt(50));
                        input.put("device_type", "phone");
                        input.put("connection_type", "wifi");
                        input.put("bid_floor", 1.0f + rand.nextFloat());
                        expectedValue = 3.0f + rand.nextFloat() * 2.0f; // $3-5
                        
                    } else if (segmentDraw < 0.20f) {
                        // Regular segment (15%)
                        input.put("os", rand.nextBoolean() ? "ios" : "android");
                        input.put("pubid", 10 + rand.nextInt(40));
                        input.put("app_bundle", "com.regular.app" + rand.nextInt(200));
                        input.put("zone_id", 50 + rand.nextInt(200));
                        input.put("device_type", rand.nextBoolean() ? "phone" : "tablet");
                        input.put("connection_type", rand.nextBoolean() ? "wifi" : "4g");
                        input.put("bid_floor", 0.1f + rand.nextFloat() * 0.5f);
                        expectedValue = 0.5f + rand.nextFloat() * 1.5f; // $0.5-2
                        
                    } else {
                        // No-bid segment (80%)
                        input.put("os", "android");
                        input.put("pubid", 50 + rand.nextInt(50));
                        input.put("app_bundle", "com.junk.app" + rand.nextInt(10000));
                        input.put("zone_id", 250 + rand.nextInt(3750));
                        input.put("device_type", "phone");
                        input.put("connection_type", "3g");
                        input.put("bid_floor", 0.01f + rand.nextFloat() * 0.1f);
                        expectedValue = -0.25f; // Penalty
                    }
                    
                    totalRequests.incrementAndGet();
                    
                    // ALWAYS predict (this is production!)
                    float prediction = model.predictFloat(input);
                    totalPredictions.incrementAndGet();
                    
                    // 2% training rate
                    if (rand.nextFloat() < 0.02f) {
                        model.train(input, expectedValue);
                        totalTrains.incrementAndGet();
                    }
                    
                    // Simulate using the prediction (e.g., deciding whether to bid)
                    boolean shouldBid = prediction > 0.1f; // Simple threshold
                });
            }
            
            // Throttle to ~5000 requests/second
            Thread.sleep(20);
        }
        
        // Shutdown
        running = false;
        executor.shutdown();
        executor.awaitTermination(10, TimeUnit.SECONDS);
        monitor.join();
        
        // Final report
        System.out.println("\n=== FINAL REPORT ===");
        System.out.printf("Total requests: %d\n", totalRequests.get());
        System.out.printf("Total trains: %d (%.1f%%)\n", 
            totalTrains.get(), 100.0 * totalTrains.get() / totalRequests.get());
        System.out.printf("Total predictions: %d\n", totalPredictions.get());
        System.out.printf("Collapse detections: %d\n", collapseCount.get());
        
        // Final stability check
        boolean stable = testFinalStability(model);
        
        if (collapseCount.get() > 0) {
            System.out.println("\n❌ FAILURE: Model collapsed during continuous operation!");
        } else if (!stable) {
            System.out.println("\n❌ FAILURE: Model is not stable after continuous operation!");
        } else {
            System.out.println("\n✓ SUCCESS: Model remained stable throughout continuous operation!");
        }
    }
    
    private void testStability(SimpleNetFloat model, Set<String> premiumApps, AtomicInteger collapseCount) {
        // Quick stability check - test a few known segments
        float premiumPred = model.predictFloat(Map.of(
            "os", "ios", "pubid", 1, "app_bundle", "com.premium.app0",
            "zone_id", 0, "device_type", "phone", "connection_type", "wifi", 
            "bid_floor", 2.0f
        ));
        
        float regularPred = model.predictFloat(Map.of(
            "os", "android", "pubid", 20, "app_bundle", "com.regular.app50",
            "zone_id", 150, "device_type", "phone", "connection_type", "4g",
            "bid_floor", 0.5f
        ));
        
        float junkPred = model.predictFloat(Map.of(
            "os", "android", "pubid", 90, "app_bundle", "com.junk.app9999",
            "zone_id", 3999, "device_type", "phone", "connection_type", "3g",
            "bid_floor", 0.01f
        ));
        
        System.out.printf("  Current predictions: Premium=$%.2f, Regular=$%.2f, No-bid=$%.2f",
            premiumPred, regularPred, junkPred);
        
        // Check for collapse
        if (Math.abs(premiumPred - regularPred) < 0.1f && 
            Math.abs(regularPred - junkPred) < 0.1f) {
            System.out.print(" ⚠️ COLLAPSE!");
            collapseCount.incrementAndGet();
        } else if (premiumPred > regularPred && regularPred > junkPred) {
            System.out.print(" ✓");
        } else {
            System.out.print(" ⚠️ Wrong order!");
        }
        System.out.println();
    }
    
    private boolean testFinalStability(SimpleNetFloat model) {
        System.out.println("\nFinal stability assessment:");
        
        // Test 20 segments of each type
        List<Float> premiumPreds = new ArrayList<>();
        List<Float> regularPreds = new ArrayList<>();
        List<Float> junkPreds = new ArrayList<>();
        
        Random rand = new Random(999);
        
        for (int i = 0; i < 20; i++) {
            // Premium
            premiumPreds.add(model.predictFloat(Map.of(
                "os", "ios", "pubid", 1 + i % 5, "app_bundle", "com.premium.app" + i,
                "zone_id", i, "device_type", "phone", "connection_type", "wifi",
                "bid_floor", 2.0f
            )));
            
            // Regular
            regularPreds.add(model.predictFloat(Map.of(
                "os", i % 2 == 0 ? "ios" : "android", "pubid", 10 + i,
                "app_bundle", "com.regular.app" + i, "zone_id", 100 + i,
                "device_type", "phone", "connection_type", "4g", "bid_floor", 0.5f
            )));
            
            // Junk
            junkPreds.add(model.predictFloat(Map.of(
                "os", "android", "pubid", 70 + i, "app_bundle", "com.junk.app" + (7000 + i),
                "zone_id", 3000 + i, "device_type", "phone", "connection_type", "3g",
                "bid_floor", 0.01f
            )));
        }
        
        // Calculate stats
        float premiumAvg = premiumPreds.stream().reduce(0f, Float::sum) / premiumPreds.size();
        float regularAvg = regularPreds.stream().reduce(0f, Float::sum) / regularPreds.size();
        float junkAvg = junkPreds.stream().reduce(0f, Float::sum) / junkPreds.size();
        
        float premiumStd = calculateStd(premiumPreds, premiumAvg);
        float regularStd = calculateStd(regularPreds, regularAvg);
        float junkStd = calculateStd(junkPreds, junkAvg);
        
        System.out.printf("Premium: avg=$%.2f (±%.2f)\n", premiumAvg, premiumStd);
        System.out.printf("Regular: avg=$%.2f (±%.2f)\n", regularAvg, regularStd);
        System.out.printf("No-bid: avg=$%.2f (±%.2f)\n", junkAvg, junkStd);
        
        // Check stability criteria
        boolean correctOrder = premiumAvg > regularAvg + 0.5f && regularAvg > junkAvg + 0.5f;
        boolean reasonableRanges = premiumAvg > 2.0f && premiumAvg < 6.0f &&
                                  regularAvg > 0.0f && regularAvg < 3.0f &&
                                  junkAvg < 0.5f;
        boolean notCollapsed = premiumStd > 0.1f || regularStd > 0.1f || junkStd > 0.1f;
        
        return correctOrder && reasonableRanges && notCollapsed;
    }
    
    private float calculateStd(List<Float> values, float mean) {
        float sum = 0;
        for (float v : values) {
            sum += (v - mean) * (v - mean);
        }
        return (float) Math.sqrt(sum / values.size());
    }
}