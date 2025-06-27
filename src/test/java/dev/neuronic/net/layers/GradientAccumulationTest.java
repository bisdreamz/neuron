package dev.neuronic.net.layers;

import dev.neuronic.net.NeuralNet;
import dev.neuronic.net.Layers;
import dev.neuronic.net.WeightInitStrategy;
import dev.neuronic.net.optimizers.SgdOptimizer;
import dev.neuronic.net.activators.LinearActivator;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests gradient accumulation functionality for true mini-batch training.
 */
public class GradientAccumulationTest {
    
    @Test
    public void testDenseLayerGradientAccumulation() {
        // Test gradient accumulation in DenseLayer directly
        SgdOptimizer optimizer = new SgdOptimizer(0.1f);
        DenseLayer layer = new DenseLayer(optimizer, 
            LinearActivator.INSTANCE, 2, 3,
            WeightInitStrategy.XAVIER);
        
        // Test that accumulation methods work
        assertTrue(layer instanceof GradientAccumulator, "DenseLayer should implement GradientAccumulator");
        
        // Start accumulation
        layer.startAccumulation();
        assertTrue(layer.isAccumulating(), "Layer should be in accumulation mode");
        
        // Create some test data
        float[] input1 = {1.0f, 2.0f, 3.0f};
        float[] input2 = {4.0f, 5.0f, 6.0f};
        float[] gradient1 = {0.1f, 0.2f};
        float[] gradient2 = {0.3f, 0.4f};
        
        // Forward passes
        Layer.LayerContext ctx1 = layer.forward(input1);
        Layer.LayerContext ctx2 = layer.forward(input2);
        
        // Accumulate gradients
        Layer.LayerContext[] stack = new Layer.LayerContext[]{ctx1};
        layer.backwardAccumulate(stack, 0, gradient1);
        
        stack[0] = ctx2;
        layer.backwardAccumulate(stack, 0, gradient2);
        
        // Apply accumulated gradients
        layer.applyAccumulatedGradients(2);
        
        assertFalse(layer.isAccumulating(), "Layer should not be accumulating after apply");
    }
    
    @Test
    public void testBatchTrainingRunsSuccessfully() {
        // Test that batch training runs without errors
        SgdOptimizer optimizer = new SgdOptimizer(0.5f);
        NeuralNet net = NeuralNet.newBuilder()
            .input(2)
            .layer(Layers.hiddenDenseRelu(3, optimizer))
            .output(Layers.outputLinearRegression(1, optimizer));
        
        // Training data
        float[][] batchInputs = {
            {1, 0},
            {0, 1},
            {1, 1},
            {0, 0}
        };
        
        float[][] batchTargets = {
            {1},
            {1},
            {2},
            {0}
        };
        
        // Train multiple times to ensure some change
        for (int i = 0; i < 10; i++) {
            net.trainBatch(batchInputs, batchTargets);
        }
        
        // Just verify it runs without errors
        float[][] predictions = net.predictBatch(batchInputs);
        assertNotNull(predictions);
        assertEquals(4, predictions.length);
    }
    
    @Test
    public void testBatchMethodsWork() {
        // Simply test that batch methods run without errors
        SgdOptimizer optimizer = new SgdOptimizer(0.1f);
        
        NeuralNet net = NeuralNet.newBuilder()
            .input(2)
            .layer(Layers.hiddenDenseRelu(3, optimizer))
            .output(Layers.outputLinearRegression(1, optimizer));
        
        // Simple training data
        float[][] inputs = {{1, 2}, {3, 4}};
        float[][] targets = {{1}, {2}};
        
        // Should run without throwing exceptions
        net.trainBatch(inputs, targets);
        
        // Should be able to predict
        float[][] predictions = net.predictBatch(inputs);
        assertEquals(2, predictions.length, "Should return predictions for all inputs");
        assertEquals(1, predictions[0].length, "Should return correct output size");
    }
    
    @Test
    public void testLargerBatchAccumulation() {
        SgdOptimizer optimizer = new SgdOptimizer(0.01f);
        NeuralNet net = NeuralNet.newBuilder()
            .input(10)
            .layer(Layers.hiddenDenseRelu(20, optimizer))
            .layer(Layers.hiddenDenseRelu(10, optimizer))
            .output(Layers.outputSoftmaxCrossEntropy(5, optimizer));
        
        // Create larger batch
        int batchSize = 32;
        float[][] batchInputs = new float[batchSize][10];
        float[][] batchTargets = new float[batchSize][5];
        
        // Fill with random data
        java.util.Random rand = new java.util.Random(42);
        for (int i = 0; i < batchSize; i++) {
            for (int j = 0; j < 10; j++) {
                batchInputs[i][j] = rand.nextFloat();
            }
            // One-hot encoded targets
            int targetClass = rand.nextInt(5);
            batchTargets[i][targetClass] = 1.0f;
        }
        
        // Train multiple batches
        float[] losses = new float[10];
        for (int epoch = 0; epoch < 10; epoch++) {
            // Get loss before training
            float totalLoss = 0;
            for (int i = 0; i < batchSize; i++) {
                float[] pred = net.predict(batchInputs[i]);
                totalLoss += crossEntropyLoss(pred, batchTargets[i]);
            }
            losses[epoch] = totalLoss / batchSize;
            
            // Train batch
            net.trainBatch(batchInputs, batchTargets);
        }
        
        // Verify loss decreases
        assertTrue(losses[9] < losses[0], "Loss should decrease with training");
    }
    
    private float crossEntropyLoss(float[] predictions, float[] targets) {
        float loss = 0;
        for (int i = 0; i < predictions.length; i++) {
            if (targets[i] > 0) {
                loss -= targets[i] * (float) Math.log(Math.max(predictions[i], 1e-7));
            }
        }
        return loss;
    }
}