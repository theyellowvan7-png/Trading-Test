import requests
import pandas as pd
from datetime import datetime

class DayTradingScanner:
    """VWAP + EMA Stack + Volume Strategy Scanner"""
    
    def __init__(self, api_key='YOUR_FINNHUB_API_KEY'):
        self.api_key = api_key
        self.base_url = 'https://finnhub.io/api/v1'
    
    def calculate_ema(self, prices, period):
        """Calculate Exponential Moving Average"""
        return pd.Series(prices).ewm(span=period, adjust=False).mean().iloc[-1]
    
    def calculate_vwap(self, prices, volumes):
        """Calculate Volume Weighted Average Price"""
        return sum(p * v for p, v in zip(prices, volumes)) / sum(volumes)
    
    def fetch_stock_data(self, symbol):
        """Fetch intraday data and calculate indicators"""
        try:
            quote_url = f"{self.base_url}/quote?symbol={symbol}&token={self.api_key}"
            quote = requests.get(quote_url).json()
            
            to_time = int(datetime.now().timestamp())
            from_time = to_time - 86400
            candle_url = f"{self.base_url}/stock/candle?symbol={symbol}&resolution=5&from={from_time}&to={to_time}&token={self.api_key}"
            candles = requests.get(candle_url).json()
            
            if candles['s'] != 'ok' or not candles['c']:
                return None
            
            prices = candles['c'][-50:]
            volumes = candles['v'][-50:]
            highs = candles['h'][-50:]
            lows = candles['l'][-50:]
            
            current_price = quote['c']
            ema9 = self.calculate_ema(prices, 9)
            ema20 = self.calculate_ema(prices, 20)
            vwap = self.calculate_vwap(prices, volumes)
            
            support = min(lows[-20:])
            resistance = max(highs[-20:])
            
            avg_volume = sum(volumes[-6:-1]) / 5
            volume_spike = volumes[-1] > avg_volume * 1.5
            ema_stack = ema9 > ema20
            above_vwap = current_price > vwap
            
            score = 50
            if above_vwap: score += 15
            if ema_stack: score += 15
            if volume_spike: score += 10
            if current_price > ema9 and current_price > ema20: score += 10
            
            return {
                'ticker': symbol,
                'price': current_price,
                'ema9': ema9,
                'ema20': ema20,
                'vwap': vwap,
                'score': score,
                'support': support,
                'resistance': resistance,
                'volume_spike': volume_spike,
                'ema_stack': ema_stack,
                'above_vwap': above_vwap,
                'risk_reward': round((resistance - current_price) / (current_price - support), 2)
            }
        except Exception as e:
            print(f"Error fetching {symbol}: {e}")
            return None
    
    def scan_market(self, symbols, min_score=50):
        """Scan multiple stocks and return signals"""
        results = []
        for symbol in symbols:
            print(f"Scanning {symbol}...")
            data = self.fetch_stock_data(symbol)
            if data and data['score'] >= min_score:
                results.append(data)
        results.sort(key=lambda x: x['score'], reverse=True)
        return results
    
    def get_trade_plan(self, stock_data, account_size=10000, risk_percent=1):
        """Generate trade plan with position sizing"""
        risk_amount = account_size * (risk_percent / 100)
        stop_distance = stock_data['price'] - stock_data['support']
        shares = int(risk_amount / stop_distance) if stop_distance > 0 else 0
        
        return {
            'ticker': stock_data['ticker'],
            'entry_price': stock_data['price'],
            'stop_loss': stock_data['support'],
            'target': stock_data['resistance'],
            'shares': shares,
            'max_loss': round(shares * stop_distance, 2),
            'target_profit': round((stock_data['resistance'] - stock_data['price']) * shares, 2),
            'risk_reward': stock_data['risk_reward']
        }

if __name__ == "__main__":
    scanner = DayTradingScanner(api_key='YOUR_FINNHUB_API_KEY')
    
    symbols = ['AAPL', 'NVDA', 'MSFT', 'TSLA', 'META', 'GOOGL', 'AMZN', 'AMD']
    
    print("🔍 Scanning market for trading opportunities...\n")
    signals = scanner.scan_market(symbols, min_score=70)
    
    print(f"\n✅ Found {len(signals)} trading signals:\n")
    
    for stock in signals[:5]:  # Top 5
        print(f"{'='*60}")
        print(f"📊 {stock['ticker']} - Score: {stock['score']}")
        print(f"   Price: ${stock['price']:.2f}")
        print(f"   VWAP: ${stock['vwap']:.2f} ({'✓ Above' if stock['above_vwap'] else '✗ Below'})")
        print(f"   EMA Stack: {'✓ Confirmed' if stock['ema_stack'] else '✗ Not Aligned'}")
        print(f"   Volume Spike: {'✓' if stock['volume_spike'] else '✗'}")
        print(f"   Risk:Reward: {stock['risk_reward']}:1")
        
        trade_plan = scanner.get_trade_plan(stock)
        print(f"\n📝 Trade Plan:")
        print(f"   Entry: ${trade_plan['entry_price']:.2f}")
        print(f"   Stop Loss: ${trade_plan['stop_loss']:.2f}")
        print(f"   Target: ${trade_plan['target']:.2f}")
        print(f"   Position: {trade_plan['shares']} shares")
        print(f"   Max Loss: ${trade_plan['max_loss']:.2f}")
        print(f"   Target Profit: ${trade_plan['target_profit']:.2f}\n")
    
    df = pd.DataFrame(signals)
    df.to_csv('trading_signals.csv', index=False)
    print(f"\n💾 Exported {len(signals)} signals to trading_signals.csv")
