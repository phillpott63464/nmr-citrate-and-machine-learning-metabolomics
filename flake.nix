{
  description = "Python Development Shell";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable"; # You can replace with stable or other branch if needed
  };

  outputs = { self, nixpkgs }:
    let
      system = "x86_64-linux"; # Change this for different architectures, like aarch64-linux for ARM systems
      pkgs = nixpkgs.legacyPackages.${system};
    in {
      devShells.${system}.default = pkgs.mkShell {
        name = "python-dev-shell";

        buildInputs = [
          pkgs.python313
          pkgs.python313Packages.virtualenv
          pkgs.python313Packages.pip
          pkgs.python313Packages.pandas
          pkgs.python313Packages.numpy
          pkgs.python313Packages.scipy
          pkgs.python313Packages.matplotlib
          pkgs.python313Packages.tqdm
          # # pkgs.python313Packages.yt-dlp
          # pkgs.python313Packages.mutagen
          # pkgs.python313Packages.tqdm
          # pkgs.ffmpeg
        ];

        nativeBuildInputs = [
          pkgs.git
          pkgs.pkg-config
        ];

        shellHook = ''
          echo "Welcome to your Python development environment!"
          
          virtualenv .venv
          source .venv/bin/activate
          pip install -r requirements.txt

          pwd

          # Optionally, run your main Python file (comment out if you don't want this)
          python graph.py

          echo "Exiting shell."

          exit
        '';
      };
    };
}
