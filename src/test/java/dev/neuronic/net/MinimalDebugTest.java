package dev.neuronic.net;

import dev.neuronic.net.layers.Feature;
import dev.neuronic.net.layers.Layer;
import dev.neuronic.net.optimizers.AdamWOptimizer;
import dev.neuronic.net.optimizers.Optimizer;
import dev.neuronic.net.simple.SimpleNet;
import dev.neuronic.net.simple.SimpleNetFloat;
import org.junit.jupiter.api.Test;

import java.util.*;

/**
 * Minimal test to debug why the network isn't learning.
 * Uses tiny feature space and extensive debugging output.
 */
public class MinimalDebugTest {
    
    @Test
    public void testMinimalLearning() {
        System.out.println("=== MINIMAL DEBUG TEST ===\n");
        
        // Tiny feature space
        Feature[] features = {
            Feature.embedding(2, 2, "OS"),        // 2 OS values, 2-dim embeddings
            Feature.embedding(5, 2, "ZONEID"),    // 5 zones, 2-dim embeddings  
            Feature.embedding(5, 2, "DOMAIN"),    // 5 domains, 2-dim embeddings
            Feature.embedding(10, 2, "PUB")       // 10 publishers, 2-dim embeddings
        };
        
        // Simple architecture
        Optimizer optimizer = new AdamWOptimizer(0.01f, 0.001f); // Higher LR for faster learning
        
        NeuralNet net = NeuralNet.newBuilder()
                .setDefaultOptimizer(optimizer)
                .layer(Layers.inputMixed(features))
                .layer(Layers.hiddenDenseRelu(16))  // Much smaller hidden layer
                .layer(Layers.hiddenDenseRelu(8))
                .output(Layers.outputLinearRegression(1));
        
        SimpleNetFloat model = SimpleNet.ofFloatRegression(net);
        
        // Print initial weights info
        System.out.println("Initial network info:");
        printNetworkInfo(net);
        
        // Define simple pattern: zone 0 = high value ($1.0), zones 1-4 = low value ($0.1)
        Map<Integer, Float> zoneValues = new HashMap<>();
        zoneValues.put(0, 1.0f);   // Premium zone
        zoneValues.put(1, 0.1f);   // Regular zones
        zoneValues.put(2, 0.1f);
        zoneValues.put(3, 0.1f);
        zoneValues.put(4, 0.1f);
        
        Random rand = new Random(42);
        
        // Training data
        List<Map<String, Object>> trainInputs = new ArrayList<>();
        List<Float> trainTargets = new ArrayList<>();
        
        // Generate training data - ensure we see all combinations
        for (int os = 0; os < 2; os++) {
            for (int zone = 0; zone < 5; zone++) {
                for (int domain = 0; domain < 5; domain++) {
                    for (int pub = 0; pub < 10; pub++) {
                        Map<String, Object> input = Map.of(
                            "OS", "os_" + os,
                            "ZONEID", "zone_" + zone,
                            "DOMAIN", "domain_" + domain,
                            "PUB", "pub_" + pub
                        );
                        
                        float target = zoneValues.get(zone);
                        
                        trainInputs.add(input);
                        trainTargets.add(target);
                    }
                }
            }
        }
        
        System.out.printf("Generated %d training samples\n", trainInputs.size());
        System.out.println("Target distribution:");
        System.out.println("  Zone 0: $1.0 (premium)");
        System.out.println("  Zones 1-4: $0.1 (regular)\n");
        
        // Shuffle training data
        List<Integer> indices = new ArrayList<>();
        for (int i = 0; i < trainInputs.size(); i++) indices.add(i);
        Collections.shuffle(indices, rand);
        
        // Train with mini-batches and debug output
        int batchSize = 50;
        int steps = 0;
        
        for (int epoch = 0; epoch < 10; epoch++) {
            System.out.printf("\n=== EPOCH %d ===\n", epoch + 1);
            
            float epochLoss = 0;
            int epochSamples = 0;
            
            for (int i = 0; i < trainInputs.size(); i += batchSize) {
                int end = Math.min(i + batchSize, trainInputs.size());
                
                List<Map<String, Object>> batchInputs = new ArrayList<>();
                List<Float> batchTargets = new ArrayList<>();
                
                for (int j = i; j < end; j++) {
                    int idx = indices.get(j);
                    batchInputs.add(trainInputs.get(idx));
                    batchTargets.add(trainTargets.get(idx));
                }
                
                // Train on batch
                model.trainBatchMaps(batchInputs, batchTargets);
                
                // Calculate batch loss for monitoring
                for (int j = 0; j < batchInputs.size(); j++) {
                    float pred = model.predictFloat(batchInputs.get(j));
                    float loss = (pred - batchTargets.get(j)) * (pred - batchTargets.get(j));
                    epochLoss += loss;
                    epochSamples++;
                }
                
                steps++;
                
                // Debug output every 10 steps
                if (steps % 10 == 0) {
                    debugPredictions(model, steps);
                }
            }
            
            // Epoch summary
            float avgLoss = epochLoss / epochSamples;
            System.out.printf("Epoch %d - Avg Loss: %.6f\n", epoch + 1, avgLoss);
            
            // Test specific examples
            System.out.println("\nTest predictions:");
            for (int zone = 0; zone < 5; zone++) {
                float pred = model.predictFloat(Map.of(
                    "OS", "os_0",
                    "ZONEID", "zone_" + zone,
                    "DOMAIN", "domain_0",
                    "PUB", "pub_0"
                ));
                System.out.printf("  Zone %d: predicted=%.3f, expected=%.3f\n", 
                    zone, pred, zoneValues.get(zone));
            }
        }
        
        // Final evaluation
        System.out.println("\n=== FINAL EVALUATION ===");
        evaluateLearning(model, zoneValues);
        
        // Print final network info
        System.out.println("\nFinal network info:");
        printNetworkInfo(net);
    }
    
    private void debugPredictions(SimpleNetFloat model, int step) {
        // Get predictions for a few test cases
        Set<String> uniquePreds = new HashSet<>();
        
        for (int zone = 0; zone < 5; zone++) {
            float pred = model.predictFloat(Map.of(
                "OS", "os_0",
                "ZONEID", "zone_" + zone,
                "DOMAIN", "domain_0",
                "PUB", "pub_0"
            ));
            uniquePreds.add(String.format("%.3f", pred));
        }
        
        System.out.printf("Step %d: %d unique predictions\n", step, uniquePreds.size());
    }
    
    private void evaluateLearning(SimpleNetFloat model, Map<Integer, Float> zoneValues) {
        // Test all zones with multiple examples
        float totalError = 0;
        int count = 0;
        
        System.out.println("Zone predictions (average over all combinations):");
        
        for (int zone = 0; zone < 5; zone++) {
            float zoneSum = 0;
            int zoneCount = 0;
            
            // Test multiple combinations for each zone
            for (int os = 0; os < 2; os++) {
                for (int domain = 0; domain < 5; domain++) {
                    for (int pub = 0; pub < 10; pub++) {
                        float pred = model.predictFloat(Map.of(
                            "OS", "os_" + os,
                            "ZONEID", "zone_" + zone,
                            "DOMAIN", "domain_" + domain,
                            "PUB", "pub_" + pub
                        ));
                        
                        zoneSum += pred;
                        zoneCount++;
                        
                        float error = Math.abs(pred - zoneValues.get(zone));
                        totalError += error;
                        count++;
                    }
                }
            }
            
            float avgPred = zoneSum / zoneCount;
            float expected = zoneValues.get(zone);
            System.out.printf("  Zone %d: avg_pred=%.3f, expected=%.3f, error=%.3f\n",
                zone, avgPred, expected, Math.abs(avgPred - expected));
        }
        
        float avgError = totalError / count;
        System.out.printf("\nAverage absolute error: %.3f\n", avgError);
        
        // Check if learning occurred
        float zone0Avg = getPredictionAverage(model, 0);
        float zone1Avg = getPredictionAverage(model, 1);
        
        boolean learned = zone0Avg > zone1Avg + 0.1f; // Zone 0 should predict higher
        System.out.printf("\nLearning successful: %s (zone0=%.3f, zone1=%.3f)\n", 
            learned, zone0Avg, zone1Avg);
    }
    
    private float getPredictionAverage(SimpleNetFloat model, int zone) {
        float sum = 0;
        int count = 0;
        
        for (int os = 0; os < 2; os++) {
            for (int domain = 0; domain < 5; domain++) {
                for (int pub = 0; pub < 10; pub++) {
                    float pred = model.predictFloat(Map.of(
                        "OS", "os_" + os,
                        "ZONEID", "zone_" + zone,
                        "DOMAIN", "domain_" + domain,
                        "PUB", "pub_" + pub
                    ));
                    sum += pred;
                    count++;
                }
            }
        }
        
        return sum / count;
    }
    
    private void printNetworkInfo(NeuralNet net) {
        Layer[] layers = net.getLayers();
        
        for (int i = 0; i < layers.length; i++) {
            Layer layer = layers[i];
            System.out.printf("  Layer %d: %s\n", i, layer.getClass().getSimpleName());
            
            // Try to get weight info if available
            if (layer instanceof dev.neuronic.net.layers.DenseLayer) {
                // Can't easily access weights without reflection or getter methods
                System.out.println("    (Dense layer - weights initialized)");
            }
        }
    }
}