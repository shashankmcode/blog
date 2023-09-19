+++
title = 'Creating Your Personal Ubuntu Server: A Comprehensive Step-by-Step Tutorial'
date = 2023-09-20T00:03:00+05:30
+++

Imagine all your photos, videos, documents, and even the music you love, all in one safe place that you control. That's what a home server does for you. Setting up a home server serves as your personal hub for digital activities within your home network. It's like having your own mini data center right at home. The primary purpose of a home server is to centralize and manage various services and resources, making your digital life more convenient and efficient.

This post offers a step-by-step guide for setting up your own home server.

### Hardware Requirements

You don't need a high-end setup to create your own home server. Here are the basic hardware requirements:

- **Computer:** Any laptop or PC will do.
- **Storage:** A hard drive or SSD with sufficient storage space for your needs.
- **Memory (RAM):** 4GB or more is recommended for smoother performance.
- **Processor:** A standard CPU will suffice.

### Prerequisites

Before you get started, make sure you have the following:

- **Ubuntu:** Ensure that Ubuntu is installed on your laptop or PC. If not, you can download and install it from the official [Ubuntu website](https://ubuntu.com/download).

Now that you have the necessary hardware and software, let's begin the setup process.

### 1 Enabling SSH on Ubuntu

To enable SSH on your Ubuntu system, follow these steps:

1. **Open the Terminal:** Press `Ctrl` + `Alt` + `T` together to open the Terminal or find it in the applications menu.

2. **Install the OpenSSH Client:** In the Terminal, run the following command to install the OpenSSH client:

   ```bash
   sudo apt install openssh-client
You may be prompted to enter your user password to confirm the installation.

### 1.1 Accessing Your Server's Terminal from Another Computer

Once you've enabled SSH on your server, you can access its terminal securely from another computer, acting as a client. Here's how:

1. **Open a Terminal on the Client Computer:** On the computer you want to use to connect to your server, open a terminal. The procedure may vary depending on your client's operating system (e.g., Windows, macOS, or Linux).

2. **Connect to the Server:** In the terminal on your client computer, use the following command to connect to your server, replacing `server_ip` with the actual IP address or hostname of your server:

   ```bash
   ssh username@server_ip
3. **Configuring SSH to Start Automatically**

   To ensure that SSH starts automatically after your server boots up, you can follow these steps:

4. **Start SSH Service:** Open a terminal on your server and run the following command to start the SSH service:

   ```bash
   sudo systemctl enable ssh
   sudo systemctl start ssh
### 1.2 Allowing SSH Access Through the Firewall

By default, the firewall on your server may block SSH for security reasons. To enable SSH access, follow these steps:



1. **Allow SSH Through the Firewall:** Run the following command to allow SSH traffic through the firewall. This command opens port 22, which is the default SSH port:

   ```bash
   sudo ufw allow ssh
### 2 Converting Ubuntu Desktop to Ubuntu Server

With the fundamental steps completed, the next task is to convert your Ubuntu desktop into an Ubuntu server environment



1. **Install Ubuntu Server Package:** Run the following command to install the Ubuntu server package:

   ```bash
   sudo apt install ubuntu-server
### 2.1 Switching to Multi-User Mode for Enhanced Performance

In some cases, you may not require a graphical desktop environment on your Ubuntu server, especially when you want to optimize system resources, such as RAM and CPU usage. To achieve this, you can switch to a multi-user target, which provides a text-based console interface for interacting with the system.

Here's how you can switch to multi-user mode:



1. **Run the Command:** Execute the following command within the terminal:

   ```bash
   sudo systemctl set-default multi-user.target
### 3 Accessing Your Server via Web Interface (Recommended: Umbrel)

Congratulations! You now have a fully functional Ubuntu server running at your home, ready to serve your needs. If you prefer an intuitive and beginner-friendly way to manage your server's console through a web interface, you have several options to choose from, such as CasaOS, Cockpit, and Umbrel.

For a user-friendly and straightforward experience, I recommend going with **Umbrel**. Here's why:

- **Simplified Management:** Umbrel provides an easy-to-navigate web interface that simplifies server management tasks.

- **Bitcoin and Lightning Node:** If you're into cryptocurrency and want to run a Bitcoin and Lightning node, Umbrel offers an all-in-one solution.

- **Community Support:** Umbrel has an active community and a wealth of resources, making it ideal for beginners.

To get started with Umbrel:

1. Visit the [Umbrel website](https://getumbrel.com/) and follow their installation instructions to set up the web-based interface on your server.

2. Once installed, access your server's console and manage it conveniently through the Umbrel web interface.

With Umbrel, you'll have an accessible and user-friendly way to interact with your Ubuntu server, making it an excellent choice for both beginners and experienced users.

Hopefully, this post has given you a starting point for your Ubuntu server journey. Your server, your choices! Explore Nextcloud (file storage), Plex (media), Bitwarden (passwords), and more. It's time to make your server work for you!