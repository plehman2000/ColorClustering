import time
import heapq
print("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n   #############################################################################\n")
print("   Welcome to our Color Clustering for Image Compression Command Line Interface!\n")
print("   #############################################################################\n")

print("""                                                                                 
                                        .                                       
                                    (@@@@@@@*                                   
                                .@@@@@@   @@@@@@                                
                             @@@@@@.   @@&   *@@@@@&                            
                         #@@@@@(   #@@*   (@@/   %@@@@@*                        
                        .@@@   ,@@%           &@@.   @@@                        
                        .@@@    #@@/         &@@.    @@@                        
                        .@@@        &@@, (@@/        @@@                        
                        .@@@           .#            @@@                        
                        .@@@                         @@@                        
                        .@@@                         @@@                        
                        .@@@                         @@@                        
                        .@@@#                      .&@@@                        
                          %@@@@@/               (@@@@@#                         
                    *@@@@@#  .&@@@@@,       *@@@@@%  .@@@@@@.                   
                .&@@@@@,#@@@@@/  *@@@@@%,@@@@@@,  %@@@@@//@@@@@%                
             %@@@@@/   (,  .&@@@@@,  #@@@@@/  /@@@@@#   *(   %@@@@@/            
         /@@@@@#   /@@# ,@@%   *@@@@@%    ,@@@@@&.  .&@&. %@@*  .&@@@@@,        
        @@@&.  ,@@&.        /@@(   (@@@, %@@@*   #@@*        .&@@.  *@@@#       
        @@@   ,@@%           #@@/   @@@, %@@*   %@@*          ,@@%   ,@@#       
        @@@       /@@(   /@@#       @@@, %@@*      .&@@.  .&@@,      ,@@#       
        @@@           #@&.          @@@, %@@*          *@@*          ,@@#       
        @@@                         @@@, %@@*                        ,@@#       
        @@@                         @@@, %@@*                        ,@@#       
        @@@                         @@@, %@@*                        ,@@#       
        @@@,                        @@@, %@@(                        (@@#       
        ,@@@@@%                 /@@@@@%   @@@@@@.                .@@@@@@        
            #@@@@@(         .@@@@@@.         *@@@@@%          %@@@@@/           
               .&@@@@@,  %@@@@@/                 #@@@@@(  (@@@@@#               
                   ,@@@@@@@#                        .&@@@@@@&.                  
                       *                                ,,  \n\n\n\n""")

path = input("  Please enter the local path of the JPG image you wish to use(searches in '/img/' folder): ")
k = input("\n   Please enter the number of clusters you wish to use: ")

print(f"\n\n    Making {k} clusters from the colorspace of the image at {path}...")

time.sleep(100)