from functools import partial

import pandas as pd

from ....utils import LazyChunk, LazyFrame, get_relative_fn, to_chunked
from ....view import TabularView, View, filter_by_keys, filter_by_keys_merged


class RfelView(View):
    """The mimic core tables, slightly post processed."""

    def __init__(self, short_name: str, deps: dict[str, list[str]], **kwargs) -> None:
        self.name = f"rfel_{short_name}"
        self.dataset = self.name
        self.deps = deps
        # Current datasets do not need transformer inter-table references
        self.trn_deps = {}
        self.parameters = get_relative_fn(f"parameters_{short_name}.yml")

        super().__init__(**kwargs)

    @to_chunked
    def ingest(self, name, **tables: LazyChunk):
        assert len(tables) == 1
        return next(iter(tables.values()))()


class ConsumerExpendituresView(RfelView):
    def __init__(self, **kwargs) -> None:
        super().__init__(
            short_name="ce",
            deps={
                "households": ["households"],
                "year": ["households"],
                "expenditures": ["expenditures"],
                "members": ["household_members"],
            },
            **kwargs,
        )

    @to_chunked
    def ingest(self, name, **tables: LazyChunk):
        match name:
            case "households":
                return pd.DataFrame(
                    index=tables["households"]()
                    .index.get_level_values("household_id")
                    .unique()
                    .astype(pd.Int64Dtype())
                )
            case "year":
                df = tables["households"]()
                new_df = df.reset_index()
                new_df['household_id'] = new_df['household_id'].astype(pd.Int64Dtype())
                new_df.index = new_df['household_id'] * 10000 + new_df['year']
                new_df.index.name = "year_id"
                new_df.rename(columns={'year': 'year_num'}, inplace=True)

                # sort by index
                new_df = new_df.sort_index()

                return new_df
            case "expenditures":
                new_df = tables["expenditures"]()
                new_df['household_id'] = new_df['household_id'].astype(pd.Int64Dtype())
                new_df['year_id'] = new_df['household_id'] * 10000 + new_df['year'].astype(pd.Int64Dtype())
                new_df = new_df.drop(columns=['year'])
                new_df.index.name = "expenditure_id"

                # sort by index and then by month
                new_df = new_df.sort_values(by=['household_id', 'month'])

                return new_df
            case "members":
                new_df = tables["household_members"]()
                new_df['household_id'] = new_df['household_id'].astype(pd.Int64Dtype())
                new_df['year_id'] = new_df['household_id'] * 10000 + new_df['year'].astype(pd.Int64Dtype())
                new_df = new_df.drop(columns=['year'])
                new_df.index.name = "member_id"
                return new_df
            case other:
                assert False, f"Table {other} not part of view {self.name}"
