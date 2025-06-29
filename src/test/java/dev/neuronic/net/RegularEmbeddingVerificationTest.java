package dev.neuronic.net;

import dev.neuronic.net.layers.Feature;
import dev.neuronic.net.optimizers.AdamWOptimizer;
import dev.neuronic.net.simple.SimpleNet;
import dev.neuronic.net.simple.SimpleNetFloat;
import org.junit.jupiter.api.Test;

import java.util.*;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Verify that regular embeddings work correctly in MixedFeatureInputLayer.
 * Also test the scenario where only 30-50 values dominate the traffic.
 */
public class RegularEmbeddingVerificationTest {
    
    @Test
    public void testRegularEmbeddingsInMixedFeatures() {
        System.out.println("=== REGULAR EMBEDDING VERIFICATION ===\n");
        
        AdamWOptimizer optimizer = new AdamWOptimizer(0.001f, 0.0f);
        
        Feature[] features = {
            Feature.oneHot(10, "format"),
            Feature.embedding(1000, 16, "domain"),  // Regular embedding
            Feature.embedding(500, 16, "zoneid"),   // Regular embedding
            Feature.passthrough("bidfloor")
        };
        
        NeuralNet net = NeuralNet.newBuilder()
            .setDefaultOptimizer(optimizer)
            .layer(Layers.inputMixed(features))
            .layer(Layers.hiddenDenseLeakyRelu(64, 0.01f))
            .output(Layers.outputLinearRegression(1));
            
        SimpleNetFloat model = SimpleNet.ofFloatRegression(net);
        
        // Test data
        Random rand = new Random(42);
        List<Map<String, Object>> testData = new ArrayList<>();
        List<Float> targets = new ArrayList<>();
        
        // Create diverse test cases
        for (int i = 0; i < 100; i++) {
            Map<String, Object> input = new HashMap<>();
            input.put("format", i % 10);
            input.put("domain", i);  // Different domains
            input.put("zoneid", i % 50);
            input.put("bidfloor", rand.nextFloat() * 5);
            testData.add(input);
            
            // Target depends on domain
            targets.add((float)(i % 10) + rand.nextFloat());
        }
        
        // Record initial predictions to compare later
        float[] initialPreds = new float[10];
        for (int i = 0; i < 10; i++) {
            initialPreds[i] = model.predictFloat(testData.get(i));
        }
        
        // Train
        for (int epoch = 0; epoch < 50; epoch++) {
            for (int i = 0; i < testData.size(); i++) {
                model.train(testData.get(i), targets.get(i));
            }
        }
        
        // Check that predictions have changed
        float totalChange = 0;
        for (int i = 0; i < 10; i++) {
            float newPred = model.predictFloat(testData.get(i));
            totalChange += Math.abs(newPred - initialPreds[i]);
        }
        System.out.println("Average prediction change: " + (totalChange / 10));
        
        // Check predictions
        float minPred = Float.MAX_VALUE, maxPred = Float.MIN_VALUE;
        for (int i = 0; i < 20; i++) {
            float pred = model.predictFloat(testData.get(i));
            float target = targets.get(i);
            System.out.printf("Domain %d: target=%.2f, pred=%.2f, error=%.2f\n", 
                            i, target, pred, Math.abs(pred - target));
            minPred = Math.min(minPred, pred);
            maxPred = Math.max(maxPred, pred);
        }
        
        System.out.printf("\nPrediction range: [%.2f, %.2f], width=%.2f\n", 
                        minPred, maxPred, maxPred - minPred);
        
        assertTrue(maxPred - minPred > 2.0f, 
                  "Embeddings should produce diverse predictions");
        assertTrue(totalChange > 1.0f,
                  "Predictions should change significantly after training");
    }
    
    @Test
    public void testSkewedDistribution() {
        System.out.println("\n\n=== SKEWED DISTRIBUTION TEST ===");
        System.out.println("Simulating 30-50 domains getting majority of traffic\n");
        
        AdamWOptimizer optimizer = new AdamWOptimizer(0.001f, 0.0f);
        
        // Test both approaches
        testSkewedWithOneHot(optimizer);
        System.out.println("\n---\n");
        testSkewedWithEmbeddings(optimizer);
    }
    
    private void testSkewedWithOneHot(AdamWOptimizer optimizer) {
        System.out.println("WITH ONE-HOT ENCODING:");
        
        Feature[] features = {
            Feature.oneHot(5000, "domain"),
            Feature.oneHot(1000, "zoneid")
        };
        
        NeuralNet net = NeuralNet.newBuilder()
            .setDefaultOptimizer(optimizer)
            .layer(Layers.inputMixed(features))
            .layer(Layers.hiddenDenseLeakyRelu(64, 0.01f))
            .output(Layers.outputLinearRegression(1));
            
        SimpleNetFloat model = SimpleNet.ofFloatRegression(net);
        
        trainAndTestSkewed(model, "one-hot");
    }
    
    private void testSkewedWithEmbeddings(AdamWOptimizer optimizer) {
        System.out.println("WITH EMBEDDINGS:");
        
        Feature[] features = {
            Feature.embedding(5000, 32, "domain"),  // 32-dim embeddings
            Feature.embedding(1000, 16, "zoneid")   // 16-dim embeddings
        };
        
        NeuralNet net = NeuralNet.newBuilder()
            .setDefaultOptimizer(optimizer)
            .layer(Layers.inputMixed(features))
            .layer(Layers.hiddenDenseLeakyRelu(64, 0.01f))
            .output(Layers.outputLinearRegression(1));
            
        SimpleNetFloat model = SimpleNet.ofFloatRegression(net);
        
        trainAndTestSkewed(model, "embeddings");
    }
    
    private void trainAndTestSkewed(SimpleNetFloat model, String type) {
        Random rand = new Random(42);
        
        // Create 40 "hot" domains that get 90% of traffic
        Set<Integer> hotDomains = new HashSet<>();
        while (hotDomains.size() < 40) {
            hotDomains.add(rand.nextInt(500));  // Hot domains in range 0-500
        }
        
        // Training distribution
        List<Map<String, Object>> trainingData = new ArrayList<>();
        List<Float> trainingTargets = new ArrayList<>();
        
        // Generate 10,000 training samples
        for (int i = 0; i < 10000; i++) {
            Map<String, Object> input = new HashMap<>();
            
            int domain;
            float target;
            
            if (rand.nextFloat() < 0.9f) {
                // 90% from hot domains
                domain = new ArrayList<>(hotDomains).get(rand.nextInt(hotDomains.size()));
                target = rand.nextFloat() * 4 + 1;  // Higher value
            } else {
                // 10% from cold domains
                domain = rand.nextInt(4000) + 1000;  // Cold domains 1000-5000
                target = -0.01f;  // Low/negative value
            }
            
            input.put("domain", domain);
            input.put("zoneid", rand.nextInt(1000));
            
            trainingData.add(input);
            trainingTargets.add(target);
        }
        
        // Train (abbreviated)
        for (int epoch = 0; epoch < 10; epoch++) {
            for (int i = 0; i < 1000; i++) {  // Sample of data
                int idx = rand.nextInt(trainingData.size());
                model.train(trainingData.get(idx), trainingTargets.get(idx));
            }
        }
        
        // Test predictions on hot vs cold domains
        System.out.println("Predictions on HOT domains:");
        float hotSum = 0;
        List<Integer> hotSample = new ArrayList<>(hotDomains);
        Collections.shuffle(hotSample);
        for (int i = 0; i < 5; i++) {
            Map<String, Object> input = new HashMap<>();
            input.put("domain", hotSample.get(i));
            input.put("zoneid", rand.nextInt(1000));
            float pred = model.predictFloat(input);
            hotSum += pred;
            System.out.printf("  Domain %d: %.3f\n", hotSample.get(i), pred);
        }
        
        System.out.println("\nPredictions on COLD domains:");
        float coldSum = 0;
        for (int i = 0; i < 5; i++) {
            Map<String, Object> input = new HashMap<>();
            int coldDomain = rand.nextInt(4000) + 1000;
            input.put("domain", coldDomain);
            input.put("zoneid", rand.nextInt(1000));
            float pred = model.predictFloat(input);
            coldSum += pred;
            System.out.printf("  Domain %d: %.3f\n", coldDomain, pred);
        }
        
        System.out.printf("\nAverage hot: %.3f, Average cold: %.3f, Difference: %.3f\n",
                        hotSum/5, coldSum/5, (hotSum-coldSum)/5);
    }
}