#!/usr/bin/env python3
"""
NSE Trading Assistant - Setup Test Script
Run this to verify your configuration before deployment
"""

import smtplib
import os
import yaml
import sys

def test_email():
    print("🧪 Testing email configuration...")
    
    try:
        # Load config
        if os.path.exists('config.yaml'):
            with open('config.yaml', 'r') as f:
                config = yaml.safe_load(f)
        else:
            print("❌ config.yaml not found")
            return False
        
        # Get credentials
        sender = os.getenv('EMAIL_SENDER') or config['email'].get('sender_email', '')
        password = os.getenv('EMAIL_PASSWORD') or config['email'].get('sender_password', '')
        
        if not sender or not password:
            print("❌ Email credentials missing")
            print("   Set EMAIL_SENDER and EMAIL_PASSWORD environment variables")
            return False
        
        # Test SMTP connection
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender, password)
        server.quit()
        
        print("✅ Email configuration successful!")
        print(f"   Sender: {sender}")
        print(f"   Target: rpwarade2@gmail.com")
        return True
        
    except Exception as e:
        print(f"❌ Email test failed: {e}")
        return False

def test_dependencies():
    print("\n🧪 Testing dependencies...")
    
    required = ['pandas', 'numpy', 'requests', 'yaml', 'nltk']
    missing = []
    
    for pkg in required:
        try:
            __import__(pkg)
            print(f"✅ {pkg}")
        except ImportError:
            print(f"❌ {pkg} - MISSING")
            missing.append(pkg)
    
    if missing:
        print(f"\n❌ Install: pip install {' '.join(missing)}")
        return False
    
    print("✅ All dependencies available")
    return True

def main():
    print("🚀 NSE Trading Assistant - Setup Test")
    print("=" * 40)
    
    tests = [test_dependencies, test_email]
    results = [test() for test in tests]
    
    print("\n" + "=" * 40)
    
    if all(results):
        print("🎉 All tests passed! Ready for deployment.")
        print("\n📝 Next steps:")
        print("1. Update config.yaml with your Gmail")
        print("2. Push to GitHub repository")
        print("3. Add GitHub secrets")
        print("4. Test GitHub Actions workflow")
    else:
        print("⚠️ Some tests failed. Fix issues before deploying.")
        sys.exit(1)

if __name__ == "__main__":
    main()