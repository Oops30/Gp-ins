If you get this error then 
E: Could not get lock /var/lib/dpkg/lock-frontend - open (11: Resource temporarily unavailable)
E: Unable to acquire the dpkg frontend lock (/var/lib/dpkg/lock-frontend), is another process using it?
Commands: 
ps aux | grep -i apt
root       3364  0.0  0.0   4504   836 ?        Ss   09:57   0:00 /bin/sh /usr/lib/apt/apt.systemd.daily install
root       3368  0.0  0.0   4504  1672 ?        S    09:57   0:00 /bin/sh /usr/lib/apt/apt.systemd.daily lock_is_held install

Do this
sudo kill -9 3364  --process id 3364
sudo kill -9 3368  --process id 3368

sudo rm /var/lib/dpkg/lock-frontend
sudo rm /var/lib/dpkg/lock
sudo dpkg --configure -a
sudo apt install whaatever you're doing
