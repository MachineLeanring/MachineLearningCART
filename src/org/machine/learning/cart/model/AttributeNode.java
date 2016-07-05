package org.machine.learning.cart.model;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * <p>
 * 构建 CART 决策树的结点
 * </p>
 * Create Date: 2016年6月22日
 * Last Modify: 2016年6月22日
 * 
 * @author <a href="http://weibo.com/u/5131020927">Q-WHai</a>
 * @see <a href="http://blog.csdn.net/lemon_tree12138">http://blog.csdn.net/lemon_tree12138</a>
 * @version 0.0.2
 */
public class AttributeNode {

    private String attributeName = null;
    private List<AttributeNode> childNodes = null;
    private String parentStatus = null; // 父节点的状态（表示的是从父节点过度到当前节点时的状态）
    private Set<String> classifys = null;
    
    private boolean isLeaf = false; // 是否是叶子节点
    
    public AttributeNode(String attributeName) {
        this.attributeName = attributeName;
    }
    
    public String getAttributeName() {
        return attributeName;
    }
    
    public List<AttributeNode> getChildNodes() {
        return childNodes;
    }
    
    public void addChildNodes(AttributeNode node) {
        if (childNodes == null) {
            childNodes = new ArrayList<>();
        }
        
        childNodes.add(node);
    }
    
    public void setParentStatus(String parentStatus) {
        this.parentStatus = parentStatus;
    }
    
    public String getParentStatus() {
        return parentStatus;
    }
    
    public Set<String> getClassifys() {
        return classifys;
    }
    
    public void addClassify(String classify) {
        if (classifys == null) {
            classifys = new HashSet<>();
        }
        
        classifys.add(classify);
    }
    
    // TODO ---------------------- leaf node -------------------------------
    
    public void setLeaf(boolean isLeaf) {
        this.isLeaf = isLeaf;
    }
    
    public boolean isLeaf() {
        return isLeaf;
    }
}
