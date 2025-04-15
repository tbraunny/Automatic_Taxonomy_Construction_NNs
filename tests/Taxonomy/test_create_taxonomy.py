from owlready2 import *
import unittest
from unittest.mock import MagicMock, patch


from src.taxonomy.create_taxonomy import createFacetedTaxonomy, TaxonomyCreator
from src.taxonomy.criteria import Criteria, SearchOperator, ValueOperator

from utils.constants import Constants as C
import traceback

from utils.owl_utils import *

class TestCreateTaxonomy(unittest.TestCase):

    def test_breathing_test(self):
        self.assertEqual(1,1)
    def test_create_taxonomy_optimizer_not_existing(self):
        onto = load_annetto_ontology("base")

        value = ValueOperator(Name="hasOptimizer",Op="has")
        op1 = SearchOperator(Name="Optimizer", Value=[value])
        criteria = Criteria(Name="Optimizer Criteria", Searchs = [op1])
        criteria.add(op1)
        tc = TaxonomyCreator(onto,[criteria])
        topnode, facetedTaxonomy, output = tc.create_taxonomy()
    

    def test_create_taxonomy_optimizer_existing(self):
        onto = load_annetto_ontology("base")

        value = ValueOperator(Name="hasTrainingOptimizer",Op="has")
        op1 = SearchOperator(Name="Optimizer", Value=[value])
        criteria = Criteria(Name="Optimizer Criteria", Searchs = [op1])
        criteria.add(op1)
        tc = TaxonomyCreator(onto,[criteria])
        topnode, facetedTaxonomy, output = tc.create_taxonomy()

    def test_create_taxonomy_layer_existing(self):
        onto = load_annetto_ontology("base")

        value = ValueOperator(Name="hasLayer",Op="has")
        op1 = SearchOperator(Name="Levels", Value=[value])
        criteria = Criteria(Name="Levels Criteria", Searchs = [op1])
        criteria.add(op1)
        tc = TaxonomyCreator(onto,[criteria])
        topnode, facetedTaxonomy, output = tc.create_taxonomy(faceted=True)
        keys = list(facetedTaxonomy[list(facetedTaxonomy.keys())[0]]['splits'].keys())
        self.assertEqual(len(keys),2)

if __name__ == "__main__":
    unittest.main()