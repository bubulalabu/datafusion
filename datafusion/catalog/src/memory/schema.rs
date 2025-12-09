// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! [`MemorySchemaProvider`]: In-memory implementations of [`SchemaProvider`].

use crate::{BatchedTableFunction, SchemaProvider, TableProvider};
use async_trait::async_trait;
use dashmap::DashMap;
use datafusion_common::{exec_err, DataFusionError};
use std::any::Any;
use std::sync::Arc;

/// Simple in-memory implementation of a schema.
#[derive(Debug)]
pub struct MemorySchemaProvider {
    tables: DashMap<String, Arc<dyn TableProvider>>,
    batched_table_functions: DashMap<String, Arc<BatchedTableFunction>>,
}

impl MemorySchemaProvider {
    /// Instantiates a new MemorySchemaProvider with an empty collection of tables.
    pub fn new() -> Self {
        Self {
            tables: DashMap::new(),
            batched_table_functions: DashMap::new(),
        }
    }
}

impl Default for MemorySchemaProvider {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl SchemaProvider for MemorySchemaProvider {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn table_names(&self) -> Vec<String> {
        self.tables
            .iter()
            .map(|table| table.key().clone())
            .collect()
    }

    async fn table(
        &self,
        name: &str,
    ) -> datafusion_common::Result<Option<Arc<dyn TableProvider>>, DataFusionError> {
        Ok(self.tables.get(name).map(|table| Arc::clone(table.value())))
    }

    fn register_table(
        &self,
        name: String,
        table: Arc<dyn TableProvider>,
    ) -> datafusion_common::Result<Option<Arc<dyn TableProvider>>> {
        if self.table_exist(name.as_str()) {
            return exec_err!("The table {name} already exists");
        }
        Ok(self.tables.insert(name, table))
    }

    fn deregister_table(
        &self,
        name: &str,
    ) -> datafusion_common::Result<Option<Arc<dyn TableProvider>>> {
        Ok(self.tables.remove(name).map(|(_, table)| table))
    }

    fn table_exist(&self, name: &str) -> bool {
        self.tables.contains_key(name)
    }

    fn batched_udtf_names(&self) -> Vec<String> {
        self.batched_table_functions
            .iter()
            .map(|f| f.key().clone())
            .collect()
    }

    fn batched_udtf(
        &self,
        name: &str,
    ) -> datafusion_common::Result<Option<Arc<BatchedTableFunction>>> {
        Ok(self
            .batched_table_functions
            .get(name)
            .map(|f| Arc::clone(f.value())))
    }

    fn register_batched_udtf(
        &self,
        name: String,
        function: Arc<BatchedTableFunction>,
    ) -> datafusion_common::Result<Option<Arc<BatchedTableFunction>>> {
        if self.batched_udtf_exist(name.as_str()) {
            return exec_err!("The batched table function {name} already exists");
        }
        Ok(self.batched_table_functions.insert(name, function))
    }

    fn deregister_batched_udtf(
        &self,
        name: &str,
    ) -> datafusion_common::Result<Option<Arc<BatchedTableFunction>>> {
        Ok(self
            .batched_table_functions
            .remove(name)
            .map(|(_, f)| f))
    }

    fn batched_udtf_exist(&self, name: &str) -> bool {
        self.batched_table_functions.contains_key(name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{BatchResultStream, BatchedTableFunctionImpl};
    use arrow::array::ArrayRef;
    use arrow::datatypes::{DataType, Field, Schema};
    use datafusion_common::Result;
    use datafusion_expr::{Expr, Signature};

    #[derive(Debug)]
    struct DummyBatchedTableFunc;

    #[async_trait::async_trait]
    impl BatchedTableFunctionImpl for DummyBatchedTableFunc {
        fn name(&self) -> &str {
            "dummy_batched_func"
        }

        fn signature(&self) -> &Signature {
            unimplemented!()
        }

        fn return_type(&self, _arg_types: &[DataType]) -> Result<Schema> {
            Ok(Schema::new(vec![Field::new(
                "value",
                DataType::Int32,
                false,
            )]))
        }

        async fn invoke_batch(
            &self,
            _args: &[ArrayRef],
            _projection: Option<&[usize]>,
            _filters: &[Expr],
            _limit: Option<usize>,
        ) -> Result<BatchResultStream> {
            unimplemented!()
        }
    }

    #[test]
    fn test_register_and_retrieve_batched_udtf() {
        let schema = MemorySchemaProvider::new();
        let func = Arc::new(BatchedTableFunction::new(Arc::new(DummyBatchedTableFunc)));

        let result = schema.register_batched_udtf("my_batched_func".to_string(), func.clone());
        assert!(result.is_ok());
        assert!(result.unwrap().is_none());

        assert!(schema.batched_udtf_exist("my_batched_func"));
        assert_eq!(schema.batched_udtf_names(), vec!["my_batched_func"]);

        let retrieved = schema.batched_udtf("my_batched_func").unwrap();
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().name(), "dummy_batched_func");
    }

    #[test]
    fn test_duplicate_batched_udtf_registration_fails() {
        let schema = MemorySchemaProvider::new();
        let func = Arc::new(BatchedTableFunction::new(Arc::new(DummyBatchedTableFunc)));

        schema
            .register_batched_udtf("my_batched_func".to_string(), func.clone())
            .unwrap();

        let result = schema.register_batched_udtf("my_batched_func".to_string(), func.clone());
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("already exists"));
    }

    #[test]
    fn test_deregister_batched_udtf() {
        let schema = MemorySchemaProvider::new();
        let func = Arc::new(BatchedTableFunction::new(Arc::new(DummyBatchedTableFunc)));

        schema
            .register_batched_udtf("my_batched_func".to_string(), func)
            .unwrap();
        assert!(schema.batched_udtf_exist("my_batched_func"));

        let removed = schema.deregister_batched_udtf("my_batched_func").unwrap();
        assert!(removed.is_some());
        assert!(!schema.batched_udtf_exist("my_batched_func"));
        assert_eq!(schema.batched_udtf_names(), Vec::<String>::new());

        let removed = schema.deregister_batched_udtf("my_batched_func").unwrap();
        assert!(removed.is_none());
    }

    #[test]
    fn test_batched_udtf_not_found() {
        let schema = MemorySchemaProvider::new();

        assert!(!schema.batched_udtf_exist("nonexistent"));
        let result = schema.batched_udtf("nonexistent").unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_multiple_batched_udtfs() {
        let schema = MemorySchemaProvider::new();
        let func1 = Arc::new(BatchedTableFunction::new(Arc::new(DummyBatchedTableFunc)));
        let func2 = Arc::new(BatchedTableFunction::new(Arc::new(DummyBatchedTableFunc)));

        schema
            .register_batched_udtf("func1".to_string(), func1)
            .unwrap();
        schema
            .register_batched_udtf("func2".to_string(), func2)
            .unwrap();

        let mut names = schema.batched_udtf_names();
        names.sort();
        assert_eq!(names, vec!["func1", "func2"]);

        assert!(schema.batched_udtf_exist("func1"));
        assert!(schema.batched_udtf_exist("func2"));
    }
}
