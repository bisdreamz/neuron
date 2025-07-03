package dev.neuronic.examples;

import dev.neuronic.net.Layers;
import dev.neuronic.net.NeuralNet;
import dev.neuronic.net.optimizers.SgdOptimizer;
import dev.neuronic.net.simple.SimpleNet;
import dev.neuronic.net.simple.SimpleNetFloat;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ThreadLocalRandom;

public class XorMapDemo {

    public static void main(String[] args) {
        NeuralNet net = NeuralNet.newBuilder()
                .setDefaultOptimizer(new SgdOptimizer(0.01f))
                .layer(Layers.inputAllNumerical(2, new String[]{ "a", "b" }))
                .layer(Layers.hiddenDenseLeakyRelu(8))
                .layer(Layers.hiddenDenseLeakyRelu(4))
                .output(Layers.outputLinearRegression(1));

        SimpleNetFloat simpleNet = SimpleNet.ofFloatRegression(net);

        List<Map<String, Object>> data = new ArrayList<>();
        List<Float> labels = new ArrayList<>();

        for (int x = 0; x < 500; x++) {
            int a = ThreadLocalRandom.current().nextInt(2);
            int b = ThreadLocalRandom.current().nextInt(2);
            int answer = a ^ b;
            data.add(Map.of("a", (float) a, "b", (float) b));
            labels.add((float) answer);
        }

        for (int i = 0; i < data.size(); i++) {
            simpleNet.train(data.get(i), labels.get(i));
        }

        System.out.println("0 and 1 " + simpleNet.predictFloat(Map.of("a", 0, "b", 1)));
        System.out.println("0 and 0 " + simpleNet.predictFloat(Map.of("a", 0, "b", 0)));
        System.out.println("1 and 0 " + simpleNet.predictFloat(Map.of("a", 1, "b", 0)));
        System.out.println("1 and 1 " + simpleNet.predictFloat(Map.of("a", 1, "b", 1)));
    }

}
