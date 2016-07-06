package org.machine.learning.cart.client;

import java.io.IOException;
import java.util.List;
import java.util.Set;

import org.core.utils.str.StringUtils;
import org.machine.learning.cart.core.CARTCore;
import org.machine.learning.cart.model.AttributeNode;
import org.machine.learning.cart.model.MinGINITuple;
import org.machine.learning.cart.utils.CARTUtils;
import org.machine.learning.cart.utils.DecisionTreeUtils;

public class CARTClient {

    public static void main(String[] args) throws IOException {
        new CARTClient().start();
    }
    
    private void start() throws IOException {
        List<List<String>> rawData = DecisionTreeUtils.getTrainingData("./data/loan.txt");
        CARTCore core = new CARTCore();
        createDecisionTree(core, rawData);
    }
    
    private void createDecisionTree(CARTCore core, List<List<String>> currentData) {
        MinGINITuple<String, String, Double> minGINITuple = core.minGiniMap(currentData);
        CARTUtils.transformAttruteStatus(currentData, minGINITuple);
        
        AttributeNode rootNode = new AttributeNode(minGINITuple.getAttriuteName());
        setAttributeNodeStatus(core, currentData, rootNode);
        DecisionTreeUtils.showDecisionTree(rootNode, "");
    }
    
    /**
     * 设置特征属性节点的分支及子节点
     * 
     * @param core
     * @param currentData
     * @param node
     */
    private void setAttributeNodeStatus(CARTCore core, List<List<String>> currentData, AttributeNode node) {
        List<String> attributeBranchList = DecisionTreeUtils.getAttributeBranchList(currentData, node.getAttributeName());
        
        int attributeIndex = DecisionTreeUtils.getAttributeIndex(currentData.get(0), node.getAttributeName());
        
        for (String attributeBranch : attributeBranchList) {
            List<List<String>> splitAttributeDataList = DecisionTreeUtils.splitAttributeDataList(currentData, attributeBranch, attributeIndex);
            buildDecisionTree(core, attributeBranch, splitAttributeDataList, node);
        }
    }
    
    /**
     * 构建 CART 决策树
     * 
     * @param core
     * @param attributeBranch
     * @param splitAttributeDataList
     * @param node
     */
    private void buildDecisionTree(CARTCore core, String attributeBranch, List<List<String>> splitAttributeDataList, AttributeNode node) {
        MinGINITuple<String, String, Double> minGINITuple = core.minGiniMap(splitAttributeDataList);
        CARTUtils.transformAttruteStatus(splitAttributeDataList, minGINITuple);
        
        // 此处为"剪枝"
        Set<String> classifySet = CARTUtils.getClassifyList(splitAttributeDataList);
        if (classifySet.size() == 1) {
            AttributeNode leafNode = new AttributeNode(classifySet.iterator().next());
            leafNode.setLeaf(true);
            leafNode.setParentStatus(attributeBranch);
            node.addChildNodes(leafNode);
            return;
        }
        
        String attributeName = minGINITuple.getAttriuteName();
        if (StringUtils.isEmpty(attributeName)) {
            List<String> singleLineData = splitAttributeDataList.get(splitAttributeDataList.size() - 1);
            AttributeNode leafNode = new AttributeNode(singleLineData.get(singleLineData.size() - 1));
            leafNode.setLeaf(true);
            leafNode.setParentStatus(attributeBranch);
            node.addChildNodes(leafNode);
            return;
        }
        
        AttributeNode attributeNode = getNewAttributeNode(attributeName, attributeBranch, node);
        setAttributeNodeStatus(core, splitAttributeDataList, attributeNode);
    }
    
    private AttributeNode getNewAttributeNode(String attributeName, String attributeBranch, AttributeNode node) {
        AttributeNode attributeNode = new AttributeNode(attributeName);
        attributeNode.setParentStatus(attributeBranch);
        node.addChildNodes(attributeNode);
        
        return attributeNode;
    }
}
