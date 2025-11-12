"""
数据契约模块：定义数据结构和验证规则
"""

from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Union
import pandas as pd

@dataclass
class DataContract:
    """数据契约基类"""
    name: str
    description: str
    required_columns: List[str]
    column_types: Dict[str, str]
    constraints: Dict[str, Any]

@dataclass
class PatientsContract(DataContract):
    """患者信息表契约"""
    def __post_init__(self):
        self.name = "patients"
        self.description = "患者基础信息表"
        self.required_columns = ["pid", "center_id"]
        self.column_types = {
            "pid": "string",
            "center_id": "string",
            "demo__age_yr": "float64",
            "demo__sex": "category"
        }
        self.constraints = {
            "pid_unique": True,
            "age_range": [18, 100],
            "sex_values": ["M", "F", "Male", "Female"]
        }

@dataclass
class SurgeryCPBContract(DataContract):
    """手术CPB信息表契约"""
    def __post_init__(self):
        self.name = "surgery_cpb"
        self.description = "手术和体外循环信息表"
        self.required_columns = ["pid"]
        self.column_types = {
            "pid": "string",
            "surg__duration_min": "float64",
            "cpb__time_min": "float64",
            "cpb__crossclamp_min": "float64"
        }
        self.constraints = {
            "pid_unique": True,
            "duration_positive": True
        }

class DataContracts:
    """数据契约管理器"""

    def __init__(self):
        self.contracts = {
            "patients": PatientsContract(),
            "surgery_cpb": SurgeryCPBContract()
        }
        self.blacklist_features = [
            "lab__*_d1", "lab__*_d2", "lab__*_d3",
            "lab__*_peak", "lab__*_low"
        ]

    def get_contract(self, table_name: str) -> Optional[DataContract]:
        """获取指定表的数据契约"""
        return self.contracts.get(table_name)

    def validate_table(self, df: pd.DataFrame, table_name: str) -> Dict[str, Any]:
        """验证表是否符合数据契约"""
        contract = self.get_contract(table_name)
        if not contract:
            return {"error": f"No contract found for table {table_name}"}

        validation_result = {
            "table_name": table_name,
            "is_valid": True,
            "errors": [],
            "warnings": []
        }

        # 检查必需列
        missing_columns = set(contract.required_columns) - set(df.columns)
        if missing_columns:
            validation_result["is_valid"] = False
            validation_result["errors"].append(f"Missing required columns: {missing_columns}")

        # 检查列类型
        for col, expected_type in contract.column_types.items():
            if col in df.columns:
                actual_type = str(df[col].dtype)
                if not self._type_compatible(actual_type, expected_type):
                    validation_result["warnings"].append(
                        f"Column {col} type mismatch: expected {expected_type}, got {actual_type}"
                    )

        # 检查约束
        validation_result["constraint_violations"] = self._check_constraints(df, contract)

        return validation_result

    def _type_compatible(self, actual: str, expected: str) -> bool:
        """检查数据类型兼容性"""
        type_mapping = {
            "string": ["object", "string"],
            "float64": ["float64", "int64", "float32", "int32"],
            "int64": ["int64", "int32"],
            "category": ["category", "object"]
        }

        compatible_types = type_mapping.get(expected, [expected])
        return actual in compatible_types

    def _check_constraints(self, df: pd.DataFrame, contract: DataContract) -> List[Dict[str, Any]]:
        """检查数据约束"""
        violations = []

        # 检查唯一性约束
        if contract.constraints.get("pid_unique") and "pid" in df.columns:
            if df["pid"].duplicated().any():
                violations.append({
                    "constraint": "pid_unique",
                    "violations": int(df["pid"].duplicated().sum())
                })

        # 检查数值范围
        if "age_range" in contract.constraints and "demo__age_yr" in df.columns:
            min_age, max_age = contract.constraints["age_range"]
            out_of_range = ((df["demo__age_yr"] < min_age) |
                           (df["demo__age_yr"] > max_age)).sum()
            if out_of_range > 0:
                violations.append({
                    "constraint": "age_range",
                    "violations": int(out_of_range),
                    "range": [min_age, max_age]
                })

        return violations

    def is_feature_blacklisted(self, feature_name: str) -> bool:
        """检查特征是否在黑名单中"""
        for blacklist_pattern in self.blacklist_features:
            if self._match_pattern(feature_name, blacklist_pattern):
                return True
        return False

    def _match_pattern(self, name: str, pattern: str) -> bool:
        """简单的模式匹配"""
        import re
        # 将通配符 * 转换为正则表达式
        regex_pattern = pattern.replace('*', '.*')
        return bool(re.match(regex_pattern, name))