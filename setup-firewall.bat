@echo off
echo Setting up Windows Firewall for TraeGPT Backend...
netsh advfirewall firewall add rule name="TraeGPT Backend" dir=in action=allow protocol=TCP localport=8000echo Firewall rule added successfully!
echo.
echo Your TraeGPT backend will be accessible at:
echo http://1009:8000
echo.
pause 