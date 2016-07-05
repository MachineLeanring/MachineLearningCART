package org.machine.learning.cart.demo;

import java.io.IOException;
import java.util.List;

import org.machine.learning.cart.core.CARTCore;
import org.machine.learning.cart.model.MinGINITuple;
import org.machine.learning.cart.utils.DecisionTreeUtils;

public class Demos {

    public static void main(String[] args) throws IOException {
        List<List<String>> currentData = DecisionTreeUtils.getTrainingData("./data/variety2.txt");
        CARTCore core = new CARTCore();
        MinGINITuple<String, String, Double> minGINITuple = core.minGiniMap(currentData);
        System.out.println(minGINITuple);
    }
}
