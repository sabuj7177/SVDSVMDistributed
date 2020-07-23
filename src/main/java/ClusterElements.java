import java.util.ArrayList;
import java.util.List;

public class ClusterElements {
    private List<Integer> elements;
    private int clusterLength;

    public ClusterElements() {
        elements = new ArrayList<>();
        clusterLength = 0;
    }

    public void addElement(int element){
        elements.add(element);
        clusterLength+=1;
    }

    public List<Integer> getElements(){
        return elements;
    }

    public int getClusterLength(){
        return clusterLength;
    }
}
