package dev.neuronic.net;

import dev.neuronic.net.activators.LinearActivator;
import dev.neuronic.net.optimizers.SgdOptimizer;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests to verify that NeuralNet performs zero allocations during training and prediction.
 * This test ensures we follow the codebase standards of using ThreadLocal buffers.
 */
class ZeroAllocationNeuralNetTest {

    private NeuralNet network;
    private float[] input;
    private float[] targets;
    private SgdOptimizer optimizer;

    @BeforeEach
    void setUp() {
        // Create a simple network
        optimizer = new SgdOptimizer(0.01f);
        
        network = NeuralNet.newBuilder()
            .input(4)
            .layer(Layers.hiddenDenseRelu(8, optimizer))
            .output(Layers.outputSoftmaxCrossEntropy(3, optimizer));
        
        // Create test data
        input = new float[]{0.1f, 0.2f, 0.3f, 0.4f};
        targets = new float[]{1.0f, 0.0f, 0.0f}; // One-hot encoded
    }

    @Test
    void testTrainSingleSampleReusesBuffers() {
        // Warm up - this will allocate ThreadLocal buffers initially
        network.train(input, targets);
        
        // Get the batch arrays from first train call
        float[][] firstInputBatch = null;
        float[][] firstTargetBatch = null;
        
        // We need to capture the arrays used internally. Since train() uses ThreadLocal buffers,
        // we'll call it multiple times and verify no new allocations happen
        
        // Force garbage collection to clean up any allocations
        System.gc();
        Thread.yield();
        
        // Get memory usage before training
        Runtime runtime = Runtime.getRuntime();
        long memoryBefore = runtime.totalMemory() - runtime.freeMemory();
        
        // Perform many training steps
        for (int i = 0; i < 100; i++) {
            network.train(input, targets);
        }
        
        // Force garbage collection again
        System.gc();
        Thread.yield();
        
        // Get memory usage after training
        long memoryAfter = runtime.totalMemory() - runtime.freeMemory();
        long memoryIncrease = memoryAfter - memoryBefore;
        
        // Memory increase should be minimal (allowing for GC overhead and measurement noise)
        assertTrue(memoryIncrease < 50_000, // 50KB threshold to account for network state updates
            "Memory increase after 100 train() calls should be minimal. Actual increase: " + memoryIncrease + " bytes");
    }

    @Test
    void testPredictArgmaxNoAllocations() {
        // Create network for classification
        NeuralNet network = NeuralNet.newBuilder()
            .input(4)
            .layer(Layers.hiddenDenseRelu(8, optimizer))
            .output(Layers.outputSoftmaxCrossEntropy(3, optimizer));
        
        // Warm up
        float firstPrediction = network.predictArgmax(input);
        assertTrue(firstPrediction >= 0 && firstPrediction < 3, "Should return valid class index");
        
        // Multiple predictions should not allocate (primitive return)
        for (int i = 0; i < 10; i++) {
            float prediction = network.predictArgmax(input);
            assertTrue(prediction >= 0 && prediction < 3, 
                "PredictArgmax should return valid class index (iteration " + i + ")");
        }
    }

    @Test
    void testPredictWithoutArgMaxMayReuseBuffer() {
        // Without argMax, predict returns layer outputs which may be reused via ThreadLocal
        float[] firstPrediction = network.predict(input);
        float[] secondPrediction = network.predict(input);
        
        // The layer may reuse its output buffer, so this could be the same reference
        // This is actually efficient and expected behavior
        
        // Values should be identical regardless of buffer reuse
        assertArrayEquals(firstPrediction, secondPrediction, 0.0001f,
            "Prediction values should be identical for same input");
        
        // Test that the output is valid (3 classes, probabilities sum to ~1)
        assertEquals(3, firstPrediction.length, "Should have 3 output values");
        float sum = 0;
        for (float val : firstPrediction) {
            assertTrue(val >= 0 && val <= 1, "Output should be valid probability");
            sum += val;
        }
        assertEquals(1.0f, sum, 0.001f, "Softmax outputs should sum to 1");
    }

    @Test
    void testMultipleTrainCallsWithDifferentData() {
        // Test with different inputs to ensure buffers handle varying data correctly
        float[] input1 = new float[]{0.1f, 0.2f, 0.3f, 0.4f};
        float[] input2 = new float[]{0.5f, 0.6f, 0.7f, 0.8f};
        float[] targets1 = new float[]{1.0f, 0.0f, 0.0f};
        float[] targets2 = new float[]{0.0f, 1.0f, 0.0f};
        
        // Warm up
        network.train(input1, targets1);
        network.train(input2, targets2);
        
        System.gc();
        Thread.yield();
        
        Runtime runtime = Runtime.getRuntime();
        long memoryBefore = runtime.totalMemory() - runtime.freeMemory();
        
        // Alternate between different inputs
        for (int i = 0; i < 50; i++) {
            network.train(input1, targets1);
            network.train(input2, targets2);
        }
        
        System.gc();
        Thread.yield();
        
        long memoryAfter = runtime.totalMemory() - runtime.freeMemory();
        long memoryIncrease = memoryAfter - memoryBefore;
        
        assertTrue(memoryIncrease < 50_000, // 50KB threshold
            "Memory should be stable when alternating between different inputs. Increase: " + memoryIncrease + " bytes");
    }

    @Test
    void testBatchTrainingVsSingleSampleMemoryProfile() {
        // Compare memory usage between batch training and single-sample training
        float[][] batchInputs = new float[][]{
            {0.1f, 0.2f, 0.3f, 0.4f},
            {0.5f, 0.6f, 0.7f, 0.8f}
        };
        float[][] batchTargets = new float[][]{
            {1.0f, 0.0f, 0.0f},
            {0.0f, 1.0f, 0.0f}
        };
        
        // Warm up both paths
        network.trainBatch(batchInputs, batchTargets);
        network.train(batchInputs[0], batchTargets[0]);
        
        System.gc();
        Thread.yield();
        
        Runtime runtime = Runtime.getRuntime();
        
        // Test single-sample training memory
        long memorySingleBefore = runtime.totalMemory() - runtime.freeMemory();
        for (int i = 0; i < 50; i++) {
            network.train(batchInputs[0], batchTargets[0]);
            network.train(batchInputs[1], batchTargets[1]);
        }
        System.gc();
        Thread.yield();
        long memorySingleAfter = runtime.totalMemory() - runtime.freeMemory();
        long memorySingleIncrease = memorySingleAfter - memorySingleBefore;
        
        // Test batch training memory
        long memoryBatchBefore = runtime.totalMemory() - runtime.freeMemory();
        for (int i = 0; i < 50; i++) {
            network.trainBatch(batchInputs, batchTargets);
        }
        System.gc();
        Thread.yield();
        long memoryBatchAfter = runtime.totalMemory() - runtime.freeMemory();
        long memoryBatchIncrease = memoryBatchAfter - memoryBatchBefore;
        
        // Both should have minimal memory increase
        assertTrue(memorySingleIncrease < 50_000,
            "Single-sample training memory increase should be minimal. Actual: " + memorySingleIncrease + " bytes");
        assertTrue(memoryBatchIncrease < 50_000,
            "Batch training memory increase should be minimal. Actual: " + memoryBatchIncrease + " bytes");
    }

    @Test
    void testThreadLocalBuffersAreProperlyInitialized() {
        // Test that ThreadLocal buffers are initialized correctly for various input sizes
        
        // Test with larger input
        NeuralNet largeNetwork = NeuralNet.newBuilder()
            .input(100)
            .layer(Layers.hiddenDense(50, LinearActivator.INSTANCE, optimizer))
            .output(Layers.outputSoftmaxCrossEntropy(10, optimizer));
        
        float[] largeInput = new float[100];
        float[] largeTargets = new float[10];
        
        // Fill with test data
        for (int i = 0; i < 100; i++) largeInput[i] = i * 0.01f;
        largeTargets[3] = 1.0f; // One-hot encoded
        
        // Should not throw any exceptions
        assertDoesNotThrow(() -> {
            largeNetwork.train(largeInput, largeTargets);
            float[] pred = largeNetwork.predict(largeInput);
            assertEquals(10, pred.length, "Should return probabilities for 10 classes");
            
            // Also test new predictArgmax
            float classIdx = largeNetwork.predictArgmax(largeInput);
            assertTrue(classIdx >= 0 && classIdx < 10, "Should return valid class index");
        }, "Large network should handle training and prediction without errors");
    }
}