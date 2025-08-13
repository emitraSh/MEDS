import numpy as np
class Node:
    def __init__(self,feature=None,threshold=None,leftSub=None,rightSub=None,info_gain=None,branch_depth=None,pureLeaf_class=None):
        ''' when feature is real value  -> {feature value <= threshold} goes to left node otherwise right
            when feature is binary there is no threshold {left: Yes : 1} and {Right: No : 0} '''
        self.feature = feature
        self.threshold = threshold

        self.leftSub = leftSub
        self.rightSub = rightSub
        self.info_gain = info_gain
        #if pure node
        self.pureLeaf_class = pureLeaf_class

        #depth within tree
        self.branch_depth = branch_depth

class DecisionTree:
    def __init__(self,max_depth=2, min_num_pure_leaf:int=2, root=None):
        #stopping conditions
        self.max_depth = max_depth # to avoid stocking in loops
        self.min_num_pure_leaf = min_num_pure_leaf # to avoid over fitting, avoiding influence of small number of data

        self.root = root

    def build_branches(self, dataset, current_depth=0, ig_method= "comb"):

        x = dataset[:,:-1]
        y = dataset[:,-1]

        if len(set(y)) != 1: #when all datas belong to the same class the node is already pure
            if current_depth >= self.max_depth and self.min_num_pure_leaf >= x.shape[0]:
                best_split = self.find_best_split(dataset,ig_method)

                if best_split["info_gain"] > 0.003:
                    left_subtree = self.build_branches(best_split["dataset_left"],current_depth= current_depth+1)
                    right_subtree = self.build_branches(best_split["dataset_right"],current_depth= current_depth+1)

                    return Node(feature= best_split["feature_index"],
                                threshold= best_split["threshold"],
                                leftSub= left_subtree,
                                rightSub= right_subtree,
                                info_gain= best_split["info_gain"],
                                branch_depth= current_depth)

        y = list(y)
        leaf_node = max(y, key=y.count)
        return Node(pureLeaf_class=leaf_node, branch_depth= current_depth)


    def find_best_split(self, dataset, ig_method):
        feature_count = dataset.shape[1]
        best_split = {}
        max_info_gain = 0
        #In order to find the best split, each feature's information gain needs to be checked and the lowest ig will be selected as the first best split
        for feature in range(feature_count):
            feature_column = dataset[:,feature]
            feature_value = np.unique(feature_column)

            #each unique value within each feature is considered as a threshold to find the best information gain
            if set(feature_value) <= {0,1}:
                dataset_left = np.array(row for row in dataset if row[feature] == 1)
                dataset_right = np.array(row for row in dataset if row[feature] == 0)
                if len(dataset_left) > 0 and len(dataset_right) > 0:
                    y, left_y, right_y = dataset[:, -1], dataset_left[:, -1], dataset_right[:, -1]
                    value_th_info_gain = self.information_gain(y, left_y, right_y, ig_method)

                    if value_th_info_gain > max_info_gain:
                        best_split["feature_index"] = feature
                        best_split["threshold"] = "1,0"
                        best_split["dataset_left"] = dataset_left
                        best_split["dataset_right"] = dataset_right
                        best_split["info_gain"] = value_th_info_gain
                        max_info_gain = value_th_info_gain
            else:
                for value_th in feature_value:
                    dataset_left = np.array(row for row in dataset if row[feature] <= value_th)
                    dataset_right = np.array(row for row in dataset if row[feature] > value_th)

                    if len(dataset_left) > 0 and len(dataset_right) > 0:
                        y, left_y, right_y = dataset[:, -1], dataset_left[:, -1], dataset_right[:, -1]
                        value_th_info_gain = self.information_gain(y, left_y, right_y, ig_method)
                        if value_th_info_gain > max_info_gain:
                            best_split["feature_index"] = feature
                            best_split["threshold"] = value_th
                            best_split["dataset_left"] = dataset_left
                            best_split["dataset_right"] = dataset_right
                            best_split["info_gain"] = value_th_info_gain
                            max_info_gain = value_th_info_gain
        return best_split


    def information_gain(self, parent, left, right, ig_method):
        weight_l = len(left) / len(parent)
        weight_r = len(right) / len(parent)
        ig = 0
        if ig_method == "gini":
            ig = self.gini_ig(parent) - ((weight_l * self.gini_ig(left)) + (weight_r * self.gini_ig(right)))
            return ig
        elif ig_method == "entropy":
            ig = self.entropy_ig(parent) - ((weight_l * self.entropy_ig(left)) + (weight_r * self.entropy_ig(right)))
            return ig
        elif ig_method == "comb":
            ig_gini = self.gini_ig(parent) - ((weight_l * self.gini_ig(left)) + (weight_r * self.gini_ig(right)))
            ig_entropy = self.entropy_ig(parent) - ((weight_l * self.entropy_ig(left)) + (weight_r * self.entropy_ig(right)))
            ig = 0.4 * ig_gini + 0.6 * ig_entropy
            return ig


    def gini_ig(self, arr):
        unique_y = np.unique(arr)
        gini = 0
        for y in unique_y:
            p_y = len(arr[arr == y]) / len(arr)
            gini += p_y ** 2
        return 1 - gini

    def entropy_ig(self, arr):
        unique_y = np.unique(arr)
        entropy = 0
        for y in unique_y:
            p_y = len(arr[arr == y]) / len(arr)
            entropy += -p_y * np.log(p_y)
        return entropy

    def print_tree(self,tree = None, indent=''):
        if not tree:
            tree = self.root
        if tree.pureLeaf_class is not None:
            return print(tree.pureLeaf_class)
        else:
            print(f"{indent}Feature[{tree.feature}] <= {tree.threshold} | Info Gain={tree.info_gain:.4f}")

            # Go left
            print(f"{indent}--> True:")
            self.print_tree(tree.leftSub, indent + indent)

            # Go right
            print(f"{indent}--> False:")
            self.print_tree(tree.rightSub, indent + indent)



    def predict(self,x_test,tree):
        predictions = [self.make_prediction(x,tree) for x in x_test]
        return predictions

    def make_prediction(self,x_test_array ,tree):
        if tree.pureLeaf_class != None: return tree.pureLeaf_class
        feature_val = x_test_array[tree.feature_index]
        if feature_val <= tree.threshold:
            return self.make_prediction(x_test_array, tree.left)
        else:
            return self.make_prediction(x_test_array, tree.right)














