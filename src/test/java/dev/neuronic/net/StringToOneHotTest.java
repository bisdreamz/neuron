package dev.neuronic.net;

import dev.neuronic.net.layers.Feature;
import dev.neuronic.net.optimizers.AdamWOptimizer;
import dev.neuronic.net.simple.SimpleNet;
import dev.neuronic.net.simple.SimpleNetFloat;
import org.junit.jupiter.api.Test;

import java.util.*;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Test to reproduce the exact issue with string inputs being used with one-hot encoding.
 * The user is passing strings like "USA", "ios", "vid" but expecting one-hot encoding.
 */
public class StringToOneHotTest {
    
    @Test
    public void testStringInputsWithOneHot() {
        System.out.println("=== STRING TO ONE-HOT CONVERSION ISSUE ===\n");
        
        // Create features as the user might have configured them
        AdamWOptimizer optimizer = new AdamWOptimizer(0.01f, 0.0001f);
        
        Feature[] features = {
            Feature.oneHot(200, "GEO"),        // Countries
            Feature.oneHot(5, "DEVCON"),       // Device connection
            Feature.oneHot(1000, "DOMAIN"),    // Domains
            Feature.oneHot(10, "OS"),          // Operating systems
            Feature.oneHot(10, "FORMAT"),      // Ad formats
            Feature.oneHot(5, "DEVTYPE"),      // Device types
            Feature.oneHot(1000, "ZONEID"),    // Zone IDs
            Feature.oneHot(1000, "COMPANYID")  // Company IDs
        };
        
        NeuralNet net = NeuralNet.newBuilder()
            .setDefaultOptimizer(optimizer)
            .layer(Layers.inputMixed(features))
            .layer(Layers.hiddenDenseLeakyRelu(128, 0.01f))
            .layer(Layers.hiddenDenseLeakyRelu(64, 0.01f))
            .output(Layers.outputLinearRegression(1));
            
        SimpleNetFloat model = SimpleNet.ofFloatRegression(net);
        
        // Test 1: What happens when we pass strings directly?
        System.out.println("TEST 1: Passing strings directly (WRONG)");
        try {
            Map<String, Object> wrongInput = new HashMap<>();
            wrongInput.put("GEO", "USA");          // STRING - WRONG!
            wrongInput.put("DEVCON", "2");          // STRING - WRONG!
            wrongInput.put("DOMAIN", "com.block.juggler");
            wrongInput.put("OS", "ios");
            wrongInput.put("FORMAT", "vid");
            wrongInput.put("DEVTYPE", "1");
            wrongInput.put("ZONEID", "FgUtQqop18uf1I2fwDie");
            wrongInput.put("COMPANYID", "990943");
            
            float pred = model.predictFloat(wrongInput);
            System.out.println("ERROR: Should have thrown exception but got prediction: " + pred);
        } catch (Exception e) {
            System.out.println("Expected error: " + e.getMessage());
        }
        
        // Test 2: Correct way - convert strings to indices
        System.out.println("\nTEST 2: Converting strings to indices (CORRECT)");
        
        // Create mappings from strings to indices
        Map<String, Map<String, Integer>> stringToIndex = createStringMappings();
        
        // Generate training data
        List<Map<String, Object>> trainingData = new ArrayList<>();
        List<Float> targets = new ArrayList<>();
        
        // Sample data based on user's example
        String[][] rawData = {
            {"USA", "2", "com.block.juggler", "ios", "vid", "1", "FgUtQqop18uf1I2fwDie", "990943"},
            {"USA", "1", "com.game.app", "android", "ban", "0", "AbCdEfGhIjKlMnOp", "123456"},
            {"CAN", "2", "com.news.reader", "ios", "nat", "1", "XyZaBcDeFgHiJkLm", "789012"},
            {"MEX", "0", "com.shop.deals", "android", "vid", "0", "QwErTyUiOpAsDfGh", "345678"},
            {"USA", "2", "com.social.chat", "ios", "ban", "1", "ZxCvBnMlKjHgFdSa", "901234"}
        };
        
        Random rand = new Random(42);
        for (String[] row : rawData) {
            Map<String, Object> input = new HashMap<>();
            input.put("GEO", getOrCreateIndex(stringToIndex, "GEO", row[0]));
            input.put("DEVCON", Integer.parseInt(row[1]));  // Already numeric
            input.put("DOMAIN", getOrCreateIndex(stringToIndex, "DOMAIN", row[2]));
            input.put("OS", getOrCreateIndex(stringToIndex, "OS", row[3]));
            input.put("FORMAT", getOrCreateIndex(stringToIndex, "FORMAT", row[4]));
            input.put("DEVTYPE", Integer.parseInt(row[5])); // Already numeric
            input.put("ZONEID", getOrCreateIndex(stringToIndex, "ZONEID", row[6]));
            input.put("COMPANYID", getOrCreateIndex(stringToIndex, "COMPANYID", row[7]));
            
            trainingData.add(input);
            targets.add(rand.nextFloat() * 5 + 1); // Random target
        }
        
        // Train
        System.out.println("Training with " + trainingData.size() + " samples...");
        for (int epoch = 0; epoch < 50; epoch++) {
            for (int i = 0; i < trainingData.size(); i++) {
                model.train(trainingData.get(i), targets.get(i));
            }
        }
        
        // Check predictions
        System.out.println("\nPredictions after training:");
        for (int i = 0; i < trainingData.size(); i++) {
            float pred = model.predictFloat(trainingData.get(i));
            System.out.printf("Sample %d: target=%.2f, pred=%.2f\n", i, targets.get(i), pred);
        }
        
        // Test 3: What happens with all identical inputs?
        System.out.println("\nTEST 3: All identical inputs");
        List<Map<String, Object>> identicalInputs = new ArrayList<>();
        for (int i = 0; i < 5; i++) {
            Map<String, Object> input = new HashMap<>();
            input.put("GEO", 0);      // Always USA (index 0)
            input.put("DEVCON", 2);   // Always 2
            input.put("DOMAIN", 0);   // Always same domain
            input.put("OS", 0);       // Always ios
            input.put("FORMAT", 0);   // Always vid
            input.put("DEVTYPE", 1);  // Always 1
            input.put("ZONEID", 0);   // Always same zone
            input.put("COMPANYID", 0); // Always same company
            identicalInputs.add(input);
        }
        
        // Train with different targets but same inputs
        for (int i = 0; i < 20; i++) {
            for (int j = 0; j < 5; j++) {
                model.train(identicalInputs.get(j), (float)(j + 1));
            }
        }
        
        // Check predictions - they should all converge to mean (3.0)
        System.out.println("Predictions for identical inputs:");
        for (int i = 0; i < 5; i++) {
            float pred = model.predictFloat(identicalInputs.get(i));
            System.out.printf("Target %.0f -> Prediction: %.3f\n", (float)(i + 1), pred);
        }
        
        System.out.println("\n=== DIAGNOSIS ===");
        System.out.println("If you're passing strings directly to one-hot features, that's the problem!");
        System.out.println("One-hot features expect INTEGER indices (0, 1, 2, ...), not strings.");
        System.out.println("You need to maintain a mapping from strings to indices.");
    }
    
    private Map<String, Map<String, Integer>> createStringMappings() {
        Map<String, Map<String, Integer>> mappings = new HashMap<>();
        mappings.put("GEO", new HashMap<>());
        mappings.put("DOMAIN", new HashMap<>());
        mappings.put("OS", new HashMap<>());
        mappings.put("FORMAT", new HashMap<>());
        mappings.put("ZONEID", new HashMap<>());
        mappings.put("COMPANYID", new HashMap<>());
        return mappings;
    }
    
    private int getOrCreateIndex(Map<String, Map<String, Integer>> mappings, 
                                 String feature, String value) {
        Map<String, Integer> featureMap = mappings.get(feature);
        return featureMap.computeIfAbsent(value, k -> featureMap.size());
    }
}