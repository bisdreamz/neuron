package dev.neuronic.net;

import dev.neuronic.net.layers.Feature;
import dev.neuronic.net.optimizers.AdamWOptimizer;
import dev.neuronic.net.optimizers.SgdOptimizer;
import dev.neuronic.net.simple.SimpleNet;
import dev.neuronic.net.simple.SimpleNetFloat;
import org.junit.jupiter.api.Test;

import java.util.*;

/**
 * Debug loss values and learning during collapse.
 * TODO: REMOVE THIS FILE AFTER DEBUGGING
 */
public class LossDebugTest {
    
    @Test
    public void debugLossAndLearning() {
        System.out.println("=== LOSS AND LEARNING DEBUG ===\n");
        
        // Simple embedding case that worked before
        Feature[] features = {
            Feature.embedding(10, 4, "item")
        };
        
        SgdOptimizer optimizer = new SgdOptimizer(0.1f);
        
        NeuralNet net = NeuralNet.newBuilder()
            .input(features.length)
            .setDefaultOptimizer(optimizer)
            .layer(Layers.inputMixed(features))
            .layer(Layers.hiddenDenseRelu(32))
            .output(Layers.outputLinearRegression(1));
            
        SimpleNetFloat model = SimpleNet.ofFloatRegression(net);
        
        System.out.println("Training simple case with loss monitoring:");
        
        // Train with different targets (like successful case)
        for (int epoch = 0; epoch < 10; epoch++) {
            float totalLoss = 0;
            int sampleCount = 0;
            
            // Multiple samples per epoch
            for (int sample = 0; sample < 4; sample++) {
                Map<String, Object> input = Map.of("item", sample);
                float target = sample * 0.5f; // 0, 0.5, 1.0, 1.5
                
                // Calculate loss before training
                float pred = model.predictFloat(input);
                float loss = (pred - target) * (pred - target);
                totalLoss += loss;
                sampleCount++;
                
                // Train
                model.train(input, target);
            }
            
            float avgLoss = totalLoss / sampleCount;
            System.out.printf("Epoch %d: Average loss = %.6f\\n", epoch, avgLoss);
            
            // Check predictions after epoch
            if (epoch % 3 == 0) {
                Set<String> uniquePreds = new HashSet<>();
                System.out.print("  Predictions: ");
                for (int i = 0; i < 4; i++) {
                    Map<String, Object> input = Map.of("item", i);
                    float pred = model.predictFloat(input);
                    uniquePreds.add(String.format("%.3f", pred));
                    System.out.printf("%.3f ", pred);
                }
                System.out.printf("(unique: %d)\\n", uniquePreds.size());
                
                if (uniquePreds.size() == 1) {
                    System.out.println("  ⚠️ COLLAPSED!");
                    break;
                }
            }
        }
        System.out.println();
    }
}