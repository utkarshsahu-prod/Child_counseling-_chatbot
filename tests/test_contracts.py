import unittest

from src.openly.contracts import CONTRACT_VERSIONS, validate_contract_freeze


class TestContractsFreeze(unittest.TestCase):
    def test_contract_versions_are_frozen(self):
        self.assertEqual(
            CONTRACT_VERSIONS,
            {
                "state_schema": "v1",
                "trace_schema": "v1",
                "domain_tree_schema": "v1",
                "cross_domain_schema": "v1",
            },
        )

    def test_contract_validation_passes(self):
        result = validate_contract_freeze()
        self.assertTrue(result.is_valid)
        self.assertEqual(result.errors, [])


if __name__ == "__main__":
    unittest.main()
