#
# Copyright 2018 Quantopian, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from copy import deepcopy
from datetime import timedelta
from functools import partial
from textwrap import dedent
import datetime
import pytest
import warnings
import sys

if sys.version_info[0] < 3:
    from StringIO import StringIO
else:
    from io import StringIO

import logbook
import toolz
from logbook import TestHandler, WARNING
from nose_parameterized import parameterized
from six import iteritems, itervalues, string_types
from six.moves import range
from testfixtures import TempDirectory

import numpy as np
import pandas as pd
import pytz
from pandas.core.common import PerformanceWarning
from trading_calendars import get_calendar, register_calendar

import zipline.api
from zipline.api import FixedSlippage
from zipline.assets import Equity, Future, Asset
from zipline.assets.continuous_futures import ContinuousFuture
from zipline.assets.synthetic import make_jagged_equity_info, make_simple_equity_info
from zipline.errors import (
    AccountControlViolation,
    CannotOrderDelistedAsset,
    IncompatibleSlippageModel,
    RegisterTradingControlPostInit,
    ScheduleFunctionInvalidCalendar,
    SetCancelPolicyPostInit,
    SymbolNotFound,
    TradingControlViolation,
    UnsupportedCancelPolicy,
    UnsupportedDatetimeFormat,
    ZeroCapitalError,
)

from zipline.finance.commission import PerShare, PerTrade
from zipline.finance.execution import LimitOrder
from zipline.finance.order import ORDER_STATUS
from zipline.finance.position import PnlRealized
from zipline.finance.trading import SimulationParameters
from zipline.finance.asset_restrictions import (
    Restriction,
    HistoricalRestrictions,
    StaticRestrictions,
    RESTRICTION_STATES,
)
from zipline.finance.controls import AssetDateBounds
from zipline.testing import (
    FakeDataPortal,
    MockDailyBarReader,
    create_daily_df_for_asset,
    create_data_portal_from_trade_history,
    create_minute_df_for_asset,
    make_test_handler,
    make_trade_data_for_asset_info,
    parameter_space,
    str_to_seconds,
    to_utc,
)
from zipline.testing import RecordBatchBlotter
import zipline.testing.fixtures as zf
from zipline.test_algorithms import (
    access_account_in_init,
    access_portfolio_in_init,
    api_algo,
    api_get_environment_algo,
    api_symbol_algo,
    handle_data_api,
    handle_data_noop,
    initialize_api,
    initialize_noop,
    noop_algo,
    record_float_magic,
    record_variables,
    call_with_kwargs,
    call_without_kwargs,
    call_with_bad_kwargs_current,
    call_with_bad_kwargs_history,
    bad_type_history_assets,
    bad_type_history_fields,
    bad_type_history_bar_count,
    bad_type_history_frequency,
    bad_type_history_assets_kwarg_list,
    bad_type_current_assets,
    bad_type_current_fields,
    bad_type_can_trade_assets,
    bad_type_is_stale_assets,
    bad_type_history_assets_kwarg,
    bad_type_history_fields_kwarg,
    bad_type_history_bar_count_kwarg,
    bad_type_history_frequency_kwarg,
    bad_type_current_assets_kwarg,
    bad_type_current_fields_kwarg,
    call_with_bad_kwargs_get_open_orders,
    call_with_good_kwargs_get_open_orders,
    call_with_no_kwargs_get_open_orders,
    empty_positions,
    no_handle_data,
)
from zipline.testing.predicates import assert_equal
from zipline.utils.api_support import ZiplineAPI
from zipline.utils.context_tricks import CallbackManager, nop_context
from zipline.utils.events import (
    date_rules,
    time_rules,
    Always,
    ComposedRule,
    Never,
    OncePerDay,
)
import zipline.utils.factory as factory

# Because test cases appear to reuse some resources.


_multiprocess_can_split_ = False


class TestCashDividendPayments(
    zf.WithMakeAlgo, zf.WithCreateBarData, zf.WithDataPortal, zf.ZiplineTestCase
):
    START_DATE = pd.Timestamp("2016-01-05", tz="UTC")
    END_DATE = ASSET_FINDER_EQUITY_END_DATE = pd.Timestamp("2016-12-31", tz="UTC")
    CREATE_BARDATA_DATA_FREQUENCY = "daily"

    ASSET_FINDER_EQUITY_SIDS = set(range(1, 9))

    SPLIT_ASSET_SID = 3
    ILLIQUID_SPLIT_ASSET_SID = 4
    MERGER_ASSET_SID = 5
    ILLIQUID_MERGER_ASSET_SID = 6
    DIVIDEND_ASSET_SID = 7
    ILLIQUID_DIVIDEND_ASSET_SID = 8
    BENCHMARK_SID = 8

    @classmethod
    def make_equity_info(cls):
        frame = super(TestCashDividendPayments, cls).make_equity_info()
        frame.loc[[1, 2], "end_date"] = pd.Timestamp("2016-12-31", tz="UTC")
        return frame

    @classmethod
    def make_splits_data(cls):
        return pd.DataFrame.from_records(
            [
                {
                    "effective_date": str_to_seconds("2016-01-06"),
                    "ratio": 0.5,
                    "sid": cls.SPLIT_ASSET_SID,
                },
                {
                    "effective_date": str_to_seconds("2016-01-07"),
                    "ratio": 0.5,
                    "sid": cls.ILLIQUID_SPLIT_ASSET_SID,
                },
            ]
        )

    @classmethod
    def make_mergers_data(cls):
        return pd.DataFrame.from_records(
            [
                {
                    "effective_date": str_to_seconds("2016-01-06"),
                    "ratio": 0.5,
                    "sid": cls.MERGER_ASSET_SID,
                },
                {
                    "effective_date": str_to_seconds("2016-01-07"),
                    "ratio": 0.6,
                    "sid": cls.ILLIQUID_MERGER_ASSET_SID,
                },
            ]
        )

    @classmethod
    def make_dividends_data(cls):
        import sys

        if sys.version_info[0] < 3:
            from StringIO import StringIO
        else:
            from io import StringIO
        data = StringIO(
            dedent(
                """
            amount,ex_date,declared_date,record_date,pay_date
            1.0,02/04/2016,01/26/2016,02/08/2016,02/11/2016
            2.0,05/05/2016,04/26/2016,05/09/2016,05/12/2016
            4.0,08/04/2016,07/26/2016,08/08/2016,08/11/2016
            8.0,11/03/2016,10/25/2016,11/07/2016,11/10/2016
        """
            )
        )
        div = pd.read_csv(
            data, parse_dates=["ex_date", "declared_date", "record_date", "pay_date"]
        )
        div["sid"] = cls.DIVIDEND_ASSET_SID
        return div

    @classmethod
    def make_adjustment_writer_equity_daily_bar_reader(cls):
        return MockDailyBarReader(
            dates=cls.trading_calendar.sessions_in_range(cls.START_DATE, cls.END_DATE)
        )

    @classmethod
    def make_equity_daily_bar_data(cls, country_code, sids):
        for sid in sids:
            asset = cls.asset_finder.retrieve_asset(sid)
            yield sid, create_daily_df_for_asset(
                cls.trading_calendar,
                asset.start_date,
                asset.end_date,
                interval=2 - sid % 2,
            )

    @classmethod
    def init_class_fixtures(cls):
        super(TestCashDividendPayments, cls).init_class_fixtures()

        cls.ASSET1 = cls.asset_finder.retrieve_asset(1)
        cls.ASSET2 = cls.asset_finder.retrieve_asset(2)
        cls.SPLIT_ASSET = cls.asset_finder.retrieve_asset(cls.SPLIT_ASSET_SID)
        cls.ILLIQUID_SPLIT_ASSET = cls.asset_finder.retrieve_asset(
            cls.ILLIQUID_SPLIT_ASSET_SID
        )
        cls.MERGER_ASSET = cls.asset_finder.retrieve_asset(cls.MERGER_ASSET_SID)
        cls.ILLIQUID_MERGER_ASSET = cls.asset_finder.retrieve_asset(
            cls.ILLIQUID_MERGER_ASSET_SID
        )
        cls.DIVIDEND_ASSET = cls.asset_finder.retrieve_asset(cls.DIVIDEND_ASSET_SID)
        cls.ILLIQUID_DIVIDEND_ASSET = cls.asset_finder.retrieve_asset(
            cls.ILLIQUID_DIVIDEND_ASSET_SID
        )
        cls.ASSETS = [cls.ASSET1, cls.ASSET2]

    def test_cash_dividends_qualified(self):

        algo_code = dedent(
            """
        from zipline.api import record, sid, date_rules
        import pandas as pd
        def initialize(context):
            context.asset = sid(7)
            context.schedule_function(
                lambda c,d: c.order(c.asset, 1), 
                date_rules.on_dates([pd.Timestamp("2016-01-21")]))

        def handle_data(context, data):
            pass
        """
        )

        algo = self.make_algo(script=algo_code)
        results = algo.run()

        self.assertEqual(
            results.pnl_realized.sum().as_dataframe.loc["long", "qualified_dividend"],
            15,
        )

    def test_cash_dividends_ordinary(self):
        algo_code = dedent(
            """
        from zipline.api import record, sid, date_rules
        import pandas as pd
        def initialize(context):
            context.asset = sid(7)
            context.schedule_function(
                lambda c,d: c.order(c.asset, 1), 
                date_rules.on_dates(["2016-02-01"]))
            context.schedule_function(
                lambda c,d: c.order(c.asset, -1),
                date_rules.on_dates(["2016-02-05"]))

        def handle_data(context, data):
            pass
        """
        )

        algo = self.make_algo(script=algo_code)
        results = algo.run()

        div = results.pnl_realized.sum().as_dataframe.loc[
            ["long", "short"], ["ordinary_dividend", "qualified_dividend"]
        ]
        assert_equal(div.values, np.array([[1, 0], [0, 0]]))

    def test_cash_dividends_close_after_exdate(self):
        algo_code = dedent(
            """
        from zipline.api import record, sid, date_rules
        import pandas as pd
        def initialize(context):
            context.asset = sid(7)
            context.schedule_function(
                lambda c,d: c.order(c.asset, 1), 
                date_rules.on_dates(["2016-01-21"]))
            context.schedule_function(
                lambda c,d: c.order(c.asset, -1),
                date_rules.on_dates(["2016-05-06"]))

        def handle_data(context, data):
            pass
        """
        )

        algo = self.make_algo(script=algo_code)
        results = algo.run()

        div = results.pnl_realized.sum().as_dataframe.loc[
            ["long", "short"], ["ordinary_dividend", "qualified_dividend"]
        ]
        assert_equal(div.values, np.array([[0, 3], [0, 0]]))

    def test_cash_dividends_ordinary_reclassified(self):
        algo_code = dedent(
            """
        from zipline.api import record, sid, date_rules
        import pandas as pd
        def initialize(context):
            context.asset = sid(7)
            context.schedule_function(
                lambda c,d: c.order(c.asset, 1), 
                date_rules.on_dates(["2016-01-21"]))
            context.schedule_function(
                lambda c,d: c.order_target_value(c.asset, 0),
                date_rules.on_dates(["2016-05-03"])) # right before ex-date

        def handle_data(context, data):
            pass
        """
        )

        algo = self.make_algo(script=algo_code)
        results = algo.run()

        div = results.pnl_realized.sum().as_dataframe.loc[
            ["long", "short"], ["ordinary_dividend", "qualified_dividend"]
        ]
        assert_equal(div.values, np.array([[0, 1], [0, 0]]))

    def test_cash_dividends_ordinary_reclassified_partial(self):
        from zipline.finance.position import PnlRealized

        algo_code = dedent(
            """
        from zipline.api import record, sid, date_rules
        import pandas as pd
        def initialize(context):
            context.asset = sid(7)
            context.schedule_function(
                lambda c,d: c.order(c.asset, 2), 
                date_rules.on_dates(["2016-01-21"]))
            
            context.schedule_function(
                lambda c,d: c.order(c.asset, -1), 
                date_rules.on_dates(["2016-02-24"]))

            context.schedule_function(
                lambda c,d: c.order_target_value(c.asset, 0),
                date_rules.on_dates(["2016-05-03"])) # right before ex-date

        def handle_data(context, data):
            pass
        """
        )

        algo = self.make_algo(script=algo_code)
        results = algo.run()

        div = (
            results.loc[:"2016-03-01", "pnl_realized"]
            .sum()
            .as_dataframe.loc[:, ["ordinary_dividend", "qualified_dividend"]]
        )
        expected = PnlRealized({"ordinary_dividend": {"long": 2}}).as_dataframe.loc[
            :, ["ordinary_dividend", "qualified_dividend"]
        ]
        assert_equal(div, expected)

        div = results.pnl_realized.sum().as_dataframe.loc[
            ["long", "short"], ["ordinary_dividend", "qualified_dividend"]
        ]
        assert_equal(div.values, np.array([[1, 1], [0, 0]]))

    def test_cash_dividend_short_position(self):
        from zipline.finance.position import PnlRealized

        algo_code = dedent(
            """
        from zipline.api import record, sid, date_rules
        import pandas as pd
        def initialize(context):
            context.asset = sid(7)
            context.schedule_function(
                lambda c,d: c.order(c.asset, -1), 
                date_rules.on_dates(["2016-02-01"]))
            # context.schedule_function(
            #     lambda c,d: c.order(c.asset, 1),
            #     date_rules.on_dates(["2016-02-05"]))

        def handle_data(context, data):
            pass
        """
        )

        algo = self.make_algo(script=algo_code)
        results = algo.run()

        expected = PnlRealized({"ordinary_dividend": {"short": -15.0}})
        expected = expected.as_dataframe[["ordinary_dividend", "qualified_dividend"]]

        div = results.pnl_realized.sum().as_dataframe[
            ["ordinary_dividend", "qualified_dividend"]
        ]
        assert_equal(*div.align(expected))


class TestPnlAccounting(zf.WithMakeAlgo, zf.ZiplineTestCase):
    START_DATE = pd.Timestamp("2006-01-03", tz="utc")
    END_DATE = START_DATE + pd.Timedelta(days=800)
    SIM_PARAMS_CAPITAL_BASE = 1000

    ASSET_FINDER_EQUITY_SIDS = (1, 133)

    SIM_PARAMS_DATA_FREQUENCY = "daily"

    @classmethod
    def make_equity_daily_bar_data(cls, country_code, sids):
        days = len(cls.equity_daily_bar_days)
        prices = np.arange(1, days + 1)
        frame = pd.DataFrame(
            {
                "open": prices,
                "high": prices,
                "low": prices,
                "close": prices,
                "volume": 100,
            },
            index=cls.equity_daily_bar_days,
        )
        return ((sid, frame) for sid in sids)

    def test_pnl_realized_frame_equal(self):
        """
        Make sure that when pnl_realized as a dataframe is an element
        of a series in the dataframe, that pd.testing.assert_frame_equal
        does not error out with KeyError: 0
        """
        from zipline.finance.position import PnlRealized

        # check that PnlRealized does not equal other pd.DataFrames
        zeros = pd.DataFrame(
            np.zeros((2, 4)),
            columns=[
                "long_term",
                "short_term",
                "qualified_dividend",
                "ordinary_dividend",
            ],
            index=["long", "short"],
        )
        pnl_realized = PnlRealized()
        self.assertFalse(pnl_realized == zeros)

        a = pd.DataFrame([PnlRealized()], columns=["pnl_realized"], index=[0])
        b = pd.DataFrame([PnlRealized()], columns=["pnl_realized"], index=[0])

        self.assertFalse(np.count_nonzero(PnlRealized().as_dataframe))
        pd.testing.assert_frame_equal(a, b)

    def test_pnl_realized(self):
        """
        Ensure that a short position is opened when a long position is
        closed out when transacting in more than the shares necessary.
        """
        from itertools import count
        from zipline.finance import slippage, commission

        # simple open close
        def initialize(context, asset):
            context.set_commission(commission.PerShare(cost=0, min_trade_cost=0))
            context.set_slippage(slippage.FixedSlippage(spread=0.0))

            context.start_date = context.get_datetime()
            context.asset = context.sid(asset)
            context.counter = count()
            # buy 1 first 4 days, close out all last day

            context.order_amounts = np.repeat(0, len(self.equity_daily_bar_days))
            context.order_amounts[0] = 2  # buy 2 lot on first day
            context.order_amounts[69] = -1  # sell 1 lot on 69th day
            context.order_amounts[-30] = -1  # close out final lot

        # runs once per day
        # reuse handle_data
        def handle_data(context, data):
            try:
                amount = context.order_amounts[next(context.counter)]

                context.order(context.asset, amount)
            except IndexError:
                pass

            context.record(
                num_positions=len(context.portfolio.positions),
                pnl=context.portfolio.pnl,
            )

        result = self.run_algorithm(
            initialize=initialize,
            handle_data=handle_data,
            asset=self.ASSET_FINDER_EQUITY_SIDS[0],
        )

        self.assertTrue("pnl_realized" in result.columns.values)
        self.assertEqual(result.pnl_realized[70].as_dataframe["short_term"].sum(), 69.0)
        self.assertEqual(result.pnl_realized[70].as_dataframe["long_term"].sum(), 0.0)
        self.assertEqual(
            result.pnl_realized[-29].as_dataframe["long_term"].sum(), 522.0
        )
        self.assertEqual(result.pnl_realized[-29].as_dataframe["short_term"].sum(), 0.0)
        self.assertEqual(result.pnl_realized[-1].as_dataframe["long_term"].sum(), 0.0)
        self.assertEqual(result.pnl_realized[-1].as_dataframe["short_term"].sum(), 0.0)

    def test_close_lots(self):
        """
        Ensure that a short position is opened when a long position is
        closed out when transacting in more than the shares necessary.
        """
        from itertools import count

        # simple open close
        def initialize(context, asset):
            context.start_date = context.get_datetime()
            context.asset = context.sid(asset)
            context.counter = count()

        # runs once per day
        # reuse handle_data
        def handle_data(context, data):
            day = next(context.counter)

            if day == 0:
                context.order(context.asset, 2, target_lots=[])

            if day == 1:
                context.order(context.asset, 3, target_lots=[])

            if day == 2:
                context.order(context.asset, 1, target_lots=[])

            if day == 3:
                # sell lot acquired on day 1
                lot = sorted(context.portfolio.positions[context.asset].lots)[1]
                context.order(context.asset, -2, target_lots=[lot])

            if day == 4:
                lot = sorted(context.portfolio.positions[context.asset].lots)[2]
                context.order(context.asset, -5, target_lots=[lot])

            lots = context.portfolio.positions[context.asset].lots

            try:
                num_lots = len(lots)
            except TypeError:
                num_lots = 0

            if lots is not None:
                context.record(
                    num_lots=num_lots, lot_amounts=[n.amount for n in sorted(lots)]
                )

        asset = self.ASSET_FINDER_EQUITY_SIDS[0]
        result = self.run_algorithm(
            initialize=initialize, handle_data=handle_data, asset=asset
        )

        # expected number of lots
        expected = [(1, 1), (2, 2), (3, 3), (4, 3), (5, 1)]
        for i, exp in expected:
            idx = result.index[i]
            self.assertEqual(exp, result.loc[idx, "num_lots"])

        expected = [
            # (0, []),
            (1, [2]),
            (2, [2, 3]),
            (3, [2, 3, 1]),
            (4, [2, 1, 1]),
            (5, [-1]),
        ]

        for i, exp in expected:
            idx = result.index[i]
            self.assertEqual(exp, result.loc[idx, "lot_amounts"])

    def test_result_save(self):
        """
        Ensure that the results file may be pickled and read back.
        Test against some issues with recursive references in the results pd.DataFrame
        That causes an error when read back.
        """
        from itertools import count
        from tempfile import TemporaryFile
        from zipline.protocol import CerealBox

        # simple open close
        def initialize(context, asset):
            context.start_date = context.get_datetime()
            context.asset = context.sid(asset)
            context.counter = count()

        def closing_rule_factory(current_price):
            def closing_rule(lot):
                return max(lot, key=lambda x: x.cost_basis - current_price)

            return closing_rule

        # runs once per day
        # reuse handle_data
        def handle_data(context, data):
            day = next(context.counter)

            current_price = data.current(context.asset, "price")
            closing_rule = closing_rule_factory(current_price)
            # orders placed
            # using closing_rule to test pickling support of functions
            if day == 0:
                context.order(
                    context.asset, 2, target_lots=[], closing_rule=closing_rule
                )

            if day == 1:
                context.order(
                    context.asset, 3, target_lots=[], closing_rule=closing_rule
                )

            if day == 2:
                context.order(
                    context.asset, 1, target_lots=[], closing_rule=closing_rule
                )

            if day == 3:
                # sell lot acquired on day 1
                lot = sorted(context.portfolio.positions[context.asset].lots)[1]
                context.order(
                    context.asset, -2, target_lots=[lot], closing_rule=closing_rule
                )

            if day == 4:
                lot = sorted(context.portfolio.positions[context.asset].lots)[2]
                context.order(
                    context.asset, -5, target_lots=[lot], closing_rule=closing_rule
                )

            lots = context.portfolio.positions[context.asset].lots

            try:
                num_lots = len(lots)
            except TypeError:
                num_lots = 0

            if lots is not None:
                context.record(
                    num_lots=num_lots, lot_amounts=[n.amount for n in sorted(lots)]
                )

        asset = self.ASSET_FINDER_EQUITY_SIDS[0]
        result = self.run_algorithm(
            initialize=initialize, handle_data=handle_data, asset=asset
        )

        tmpfile = TemporaryFile()
        result.to_pickle(tmpfile)

        readback = pd.read_pickle(tmpfile)
        pd.testing.assert_frame_equal(result, readback)
        pd.testing.assert_series_equal(result.pnl_realized, readback.pnl_realized)


class TestPositions(zf.WithMakeAlgo, zf.ZiplineTestCase):
    START_DATE = pd.Timestamp("2006-01-03", tz="utc")
    END_DATE = pd.Timestamp("2006-01-06", tz="utc")
    SIM_PARAMS_CAPITAL_BASE = 1000

    ASSET_FINDER_EQUITY_SIDS = (1, 133)

    SIM_PARAMS_DATA_FREQUENCY = "daily"

    @classmethod
    def make_equity_daily_bar_data(cls, country_code, sids):
        frame = pd.DataFrame(
            {
                "open": [90, 95, 100, 105],
                "high": [90, 95, 100, 105],
                "low": [90, 95, 100, 105],
                "close": [90, 95, 100, 105],
                "volume": 100,
            },
            index=cls.equity_daily_bar_days,
        )
        return ((sid, frame) for sid in sids)

    @classmethod
    def make_futures_info(cls):
        return pd.DataFrame.from_dict(
            {
                1000: {
                    "symbol": "CLF06",
                    "root_symbol": "CL",
                    "start_date": cls.START_DATE,
                    "end_date": cls.END_DATE,
                    "auto_close_date": cls.END_DATE + cls.trading_calendar.day,
                    "exchange": "CMES",
                    "multiplier": 100,
                }
            },
            orient="index",
        )

    @classmethod
    def make_future_minute_bar_data(cls):
        trading_calendar = cls.trading_calendars[Future]

        sids = cls.asset_finder.futures_sids
        minutes = trading_calendar.minutes_for_sessions_in_range(
            cls.future_minute_bar_days[0], cls.future_minute_bar_days[-1]
        )
        frame = pd.DataFrame(
            {"open": 2.0, "high": 2.0, "low": 2.0, "close": 2.0, "volume": 100},
            index=minutes,
        )
        return ((sid, frame) for sid in sids)

    def test_portfolio_exited_position(self):
        # This test ensures ensures that 'phantom' positions do not appear in
        # context.portfolio.positions in the case that a position has been
        # entered and fully exited.

        def initialize(context, sids):
            context.ordered = False
            context.exited = False
            context.sids = sids

        def handle_data(context, data):
            if not context.ordered:
                for s in context.sids:
                    context.order(context.sid(s), 1)
                context.ordered = True

            if not context.exited:
                amounts = [
                    pos.amount for pos in itervalues(context.portfolio.positions)
                ]

                if len(amounts) > 0 and all([(amount == 1) for amount in amounts]):
                    for stock in context.portfolio.positions:
                        context.order(context.sid(stock), -1)
                    context.exited = True

            # Should be 0 when all positions are exited.
            context.record(num_positions=len(context.portfolio.positions))

        result = self.run_algorithm(
            initialize=initialize,
            handle_data=handle_data,
            sids=self.ASSET_FINDER_EQUITY_SIDS,
        )

        expected_position_count = [
            0,  # Before entering the first position
            2,  # After entering, exiting on this date
            0,  # After exiting
            0,
        ]
        for i, expected in enumerate(expected_position_count):
            self.assertEqual(result.ix[i]["num_positions"], expected)

    def test_position_lots_oversell(self):
        """
        Ensure that a short position is opened when a long position is
        closed out when transacting in more than the shares necessary.
        """

        def initialize(context, asset):
            context.start_date = context.get_datetime()
            context.ordered = False
            context.exited = False
            context.asset = context.sid(asset)

        def handle_data(context, data):
            # open the initial position
            if not context.ordered:
                context.order(context.asset, 1)
                context.ordered = True

            # wait a couple days
            if not context.exited:  # when to cut off the simulation
                if context.get_datetime() > context.start_date + pd.Timedelta(days=2):
                    context.order(context.asset, -5)
                    context.exited = True
                    context.transaction_date = context.get_datetime()

            context.record(
                num_positions=len(context.portfolio.positions),
                amount=context.portfolio.positions[context.asset].amount,
                lots=context.portfolio.positions[context.asset].lots,
            )

        result = self.run_algorithm(
            initialize=initialize,
            handle_data=handle_data,
            asset=self.ASSET_FINDER_EQUITY_SIDS[0],
        )

        last_day = result.loc[result.index[-1]]
        self.assertEqual(last_day["amount"], -4)
        self.assertEqual(len(last_day["lots"]), 1)

    def test_noop_orders(self):
        asset = self.asset_finder.retrieve_asset(1)

        # Algorithm that tries to buy with extremely low stops/limits and tries
        # to sell with extremely high versions of same. Should not end up with
        # any positions for reasonable data.
        def handle_data(algo, data):

            ########
            # Buys #
            ########

            # Buy with low limit, shouldn't trigger.
            algo.order(asset, 100, limit_price=1)

            # But with high stop, shouldn't trigger
            algo.order(asset, 100, stop_price=10000000)

            # Buy with high limit (should trigger) but also high stop (should
            # prevent trigger).
            algo.order(asset, 100, limit_price=10000000, stop_price=10000000)

            # Buy with low stop (should trigger), but also low limit (should
            # prevent trigger).
            algo.order(asset, 100, limit_price=1, stop_price=1)

            #########
            # Sells #
            #########

            # Sell with high limit, shouldn't trigger.
            algo.order(asset, -100, limit_price=1000000)

            # Sell with low stop, shouldn't trigger.
            algo.order(asset, -100, stop_price=1)

            # Sell with low limit (should trigger), but also high stop (should
            # prevent trigger).
            algo.order(asset, -100, limit_price=1000000, stop_price=1000000)

            # Sell with low limit (should trigger), but also low stop (should
            # prevent trigger).
            algo.order(asset, -100, limit_price=1, stop_price=1)

            ###################
            # Rounding Checks #
            ###################
            algo.order(asset, 100, limit_price=0.00000001)
            algo.order(asset, -100, stop_price=0.00000001)

        daily_stats = self.run_algorithm(handle_data=handle_data)

        # Verify that positions are empty for all dates.
        empty_positions = daily_stats.positions.map(lambda x: len(x) == 0)
        self.assertTrue(empty_positions.all())

    def test_position_weights(self):
        sids = (1, 133, 1000)
        equity_1, equity_133, future_1000 = self.asset_finder.retrieve_all(sids)

        def initialize(algo, sids_and_amounts, *args, **kwargs):
            algo.ordered = False
            algo.sids_and_amounts = sids_and_amounts
            algo.set_commission(us_equities=PerTrade(0), us_futures=PerTrade(0))
            algo.set_slippage(us_equities=FixedSlippage(0), us_futures=FixedSlippage(0))

        def handle_data(algo, data):
            if not algo.ordered:
                for s, amount in algo.sids_and_amounts:
                    algo.order(algo.sid(s), amount)
                algo.ordered = True

            algo.record(position_weights=algo.portfolio.current_portfolio_weights)

        daily_stats = self.run_algorithm(
            sids_and_amounts=zip(sids, [2, -1, 1]),
            initialize=initialize,
            handle_data=handle_data,
        )

        expected_position_weights = [
            # No positions held on the first day.
            pd.Series({}),
            # Each equity's position value is its price times the number of
            # shares held. In this example, we hold a long position in 2 shares
            # of equity_1 so its weight is (95.0 * 2) = 190.0 divided by the
            # total portfolio value. The total portfolio value is the sum of
            # cash ($905.00) plus the value of all equity positions.
            #
            # For a futures contract, its weight is the unit price times number
            # of shares held times the multiplier. For future_1000, this is
            # (2.0 * 1 * 100) = 200.0 divided by total portfolio value.
            pd.Series(
                {
                    equity_1: 190.0 / (190.0 - 95.0 + 905.0),
                    equity_133: -95.0 / (190.0 - 95.0 + 905.0),
                    future_1000: 200.0 / (190.0 - 95.0 + 905.0),
                }
            ),
            pd.Series(
                {
                    equity_1: 200.0 / (200.0 - 100.0 + 905.0),
                    equity_133: -100.0 / (200.0 - 100.0 + 905.0),
                    future_1000: 200.0 / (200.0 - 100.0 + 905.0),
                }
            ),
            pd.Series(
                {
                    equity_1: 210.0 / (210.0 - 105.0 + 905.0),
                    equity_133: -105.0 / (210.0 - 105.0 + 905.0),
                    future_1000: 200.0 / (210.0 - 105.0 + 905.0),
                }
            ),
        ]

        for i, expected in enumerate(expected_position_weights):
            assert_equal(daily_stats.iloc[i]["position_weights"], expected)

    def test_same_day_orders(self):
        """
        Ensure that a short position is opened when a long position is
        closed out when transacting in more than the shares necessary.
        """

        def initialize(context, asset):
            context.start_date = context.get_datetime()
            context.ordered = False
            context.exited = False
            context.asset = context.sid(asset)

        def handle_data(context, data):
            # open the initial position
            if not context.ordered:
                context.order(context.asset, 1)
                context.order(context.asset, 2)
                context.ordered = True

            context.record(
                num_positions=len(context.portfolio.positions),
                amount=context.portfolio.positions[context.asset].amount,
                cost_basis=context.portfolio.positions[context.asset].cost_basis,
                lots=context.portfolio.positions[context.asset].lots,
            )

        asset = self.ASSET_FINDER_EQUITY_SIDS[0]
        result = self.run_algorithm(
            initialize=initialize, handle_data=handle_data, asset=asset
        )
        result.to_pickle("/tmp/result.pkl")

        idx = result.index[-1]
        last_day = result.loc[idx]
        lot = next(iter(last_day["lots"]))

        self.assertEqual(len(last_day["lots"]), 1)
        self.assertEqual(lot.amount, 3)
        self.assertEqual(last_day.cost_basis, lot.cost_basis)


class TestStockDividendPayments(
    zf.WithMakeAlgo, zf.WithCreateBarData, zf.WithDataPortal, zf.ZiplineTestCase
):
    START_DATE = pd.Timestamp("2016-01-05", tz="UTC")
    END_DATE = ASSET_FINDER_EQUITY_END_DATE = pd.Timestamp("2016-12-31", tz="UTC")
    CREATE_BARDATA_DATA_FREQUENCY = "daily"

    ASSET_FINDER_EQUITY_SIDS = set(range(1, 9))

    SPLIT_ASSET_SID = 3
    ILLIQUID_SPLIT_ASSET_SID = 4
    MERGER_ASSET_SID = 5
    ILLIQUID_MERGER_ASSET_SID = 6
    DIVIDEND_ASSET_SID = 7
    ILLIQUID_DIVIDEND_ASSET_SID = 8
    STOCK_DIVIDEND_ASSET_SID = 7
    BENCHMARK_SID = 8

    @classmethod
    def make_equity_info(cls):
        frame = super(TestStockDividendPayments, cls).make_equity_info()
        frame.loc[[1, 2], "end_date"] = pd.Timestamp("2016-12-31", tz="UTC")
        return frame

    @classmethod
    def make_splits_data(cls):
        return pd.DataFrame.from_records(
            [
                {
                    "effective_date": str_to_seconds("2016-01-06"),
                    "ratio": 0.5,
                    "sid": cls.SPLIT_ASSET_SID,
                },
                {
                    "effective_date": str_to_seconds("2016-01-07"),
                    "ratio": 0.5,
                    "sid": cls.ILLIQUID_SPLIT_ASSET_SID,
                },
            ]
        )

    @classmethod
    def make_mergers_data(cls):
        return pd.DataFrame.from_records(
            [
                {
                    "effective_date": str_to_seconds("2016-01-06"),
                    "ratio": 0.5,
                    "sid": cls.MERGER_ASSET_SID,
                },
                {
                    "effective_date": str_to_seconds("2016-01-07"),
                    "ratio": 0.6,
                    "sid": cls.ILLIQUID_MERGER_ASSET_SID,
                },
            ]
        )

    # @classmethod
    # def make_dividends_data(cls):
    #     data = StringIO(
    #         dedent(
    #             """
    #         amount,ex_date,declared_date,record_date,pay_date
    #         1.0,02/04/2016,01/26/2016,02/08/2016,02/11/2016
    #         2.0,05/05/2016,04/26/2016,05/09/2016,05/12/2016
    #         4.0,08/04/2016,07/26/2016,08/08/2016,08/11/2016
    #         8.0,11/03/2016,10/25/2016,11/07/2016,11/10/2016
    #     """
    #         )
    #     )
    #     div = pd.read_csv(
    #         data, parse_dates=["ex_date", "declared_date", "record_date", "pay_date"]
    #     )
    #     div["sid"] = cls.DIVIDEND_ASSET_SID
    #     return div

    @classmethod
    def make_stock_dividends_data(cls):
        data = StringIO(
            dedent(
                """
            ratio,ex_date,declared_date,record_date,pay_date
            0.1111111,02/04/2016,01/26/2016,02/08/2016,02/11/2016
            0.2222222,05/05/2016,04/26/2016,05/09/2016,05/12/2016
            0.4444444,08/04/2016,07/26/2016,08/08/2016,08/11/2016
            0.8888888,11/03/2016,10/25/2016,11/07/2016,11/10/2016
        """
            )
        )
        div = pd.read_csv(
            data, parse_dates=["ex_date", "declared_date", "record_date", "pay_date"]
        )
        div["sid"] = cls.STOCK_DIVIDEND_ASSET_SID
        div["payment_sid"] = cls.STOCK_DIVIDEND_ASSET_SID
        return div

    @classmethod
    def make_adjustment_writer_equity_daily_bar_reader(cls):
        return MockDailyBarReader(
            dates=cls.trading_calendar.sessions_in_range(cls.START_DATE, cls.END_DATE)
        )

    @classmethod
    def make_equity_daily_bar_data(cls, country_code, sids):
        for sid in sids:
            asset = cls.asset_finder.retrieve_asset(sid)
            yield sid, create_daily_df_for_asset(
                cls.trading_calendar,
                asset.start_date,
                asset.end_date,
                interval=2 - sid % 2,
            )

    @classmethod
    def init_class_fixtures(cls):
        super(TestStockDividendPayments, cls).init_class_fixtures()

        cls.ASSET1 = cls.asset_finder.retrieve_asset(1)
        cls.ASSET2 = cls.asset_finder.retrieve_asset(2)
        cls.SPLIT_ASSET = cls.asset_finder.retrieve_asset(cls.SPLIT_ASSET_SID)
        cls.ILLIQUID_SPLIT_ASSET = cls.asset_finder.retrieve_asset(
            cls.ILLIQUID_SPLIT_ASSET_SID
        )
        cls.MERGER_ASSET = cls.asset_finder.retrieve_asset(cls.MERGER_ASSET_SID)
        cls.ILLIQUID_MERGER_ASSET = cls.asset_finder.retrieve_asset(
            cls.ILLIQUID_MERGER_ASSET_SID
        )
        cls.DIVIDEND_ASSET = cls.asset_finder.retrieve_asset(cls.DIVIDEND_ASSET_SID)
        cls.ILLIQUID_DIVIDEND_ASSET = cls.asset_finder.retrieve_asset(
            cls.ILLIQUID_DIVIDEND_ASSET_SID
        )
        cls.STOCK_DIVIDEND_ASSET = cls.asset_finder.retrieve_asset(
            cls.STOCK_DIVIDEND_ASSET_SID
        )
        cls.ASSETS = [cls.ASSET1, cls.ASSET2]

    def test_stock_dividend_payout(self):

        algo_code = dedent(
            """
        from zipline.api import record, sid, date_rules
        import pandas as pd
        def initialize(context):
            context.asset = sid({})
            context.schedule_function(
                lambda c,d: c.order(c.asset, 1e3), 
                date_rules.on_dates([pd.Timestamp("2016-01-21")]))

        def handle_data(context, data):
            pass
        """
        ).format(type(self).STOCK_DIVIDEND_ASSET_SID)

        algo = self.make_algo(script=algo_code)
        results = algo.run()

        positions = results.positions.iloc[-1][0]
        dividends = results.pnl_realized.sum().as_dataframe

        self.assertEqual(positions["amount"], 3701)
        assert_equal(
            dividends, PnlRealized(("long", "ordinary_dividend", 84255.0)).as_dataframe
        )

    def test_stock_dividend_payout_short_position(self):

        algo_code = dedent(
            """
        from zipline.api import record, sid, date_rules
        import pandas as pd
        def initialize(context):
            context.asset = sid({})
            context.schedule_function(
                lambda c,d: c.order(c.asset, -1000), 
                date_rules.on_dates([pd.Timestamp("2016-01-21")]))

        def handle_data(context, data):
            pass
        """
        ).format(type(self).STOCK_DIVIDEND_ASSET_SID)

        algo = self.make_algo(script=algo_code)
        results = algo.run()

        positions = results.positions.iloc[-1][0]
        dividends = results.pnl_realized.sum().as_dataframe

        self.assertEqual(positions["amount"], -3701)
        assert_equal(
            dividends,
            PnlRealized(("short", "ordinary_dividend", -84255.0)).as_dataframe,
        )

    def test_stock_dividend_payout_after_position_closed(self):

        algo_code = dedent(
            """
        from zipline.api import record, sid, date_rules
        import pandas as pd
        def initialize(context):
            context.asset = sid({})
            context.schedule_function(
                lambda c,d: c.order(c.asset, 1e3), 
                date_rules.on_dates(["2016-01-21"]))
            
            context.schedule_function(
                lambda c,d: c.order_target_value(c.asset, 0),
                date_rules.on_dates(["2016-02-10"]))

            # close out the stock dividends received
            context.schedule_function(
                lambda c,d: c.order_target_value(c.asset, 0),
                date_rules.on_dates(["2016-02-17"]))

        def handle_data(context, data):
            pass
        """
        ).format(type(self).STOCK_DIVIDEND_ASSET_SID)

        algo = self.make_algo(script=algo_code)
        results = algo.run()

        idx = "2016-02-16"
        positions = results.loc[idx, 'positions'][0][0]
        self.assertEqual(positions["amount"], 111)
        self.assertEqual(positions["transaction_date"].strftime("%Y-%m-%d"), "2016-01-21")

        positions = results['positions'][-1]
        self.assertEqual(positions, [])
